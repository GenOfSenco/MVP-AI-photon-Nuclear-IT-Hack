import os
import hashlib
import io
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from .model import InitNet, UpdateNet, OpticalUpdates, PhysicsFeatures, infer, optical_features
from .phase_png import encode, decode
from .physics import Fresnel, N, DEPTHS, WAVELENGTH, PITCH
from .metrics import evaluate
from .config import FINAL_BUDGET, FINAL_PRESET
from .refinement import refine





WEIGHTS = Path(__file__).resolve().parents[1] / 'weights'
MAX_PNG = 256 * 1024






class InputError(ValueError):
    """чек ввода"""
def png_array(raw, plane =None):
    prefix= f'Плоскость {plane}. ' if plane is not None else ''
    if len(raw) > MAX_PNG:
        raise InputError(prefix + 'Размер файла превышает 256 КиБ.')
    if len(raw)< 33 or raw[:16 ] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR':
        raise InputError(prefix + 'Требуется корректный PNG.')
    if int.from_bytes(raw[16:20] , 'big') != N or int.from_bytes(raw[20:24], 'big') != N:
        raise InputError(prefix +'Требуется изображение ровно 128×128.')
    if raw[24:26]!= b'\x08\x00':
        raise InputError(prefix + 'Требуется 8-битный grayscale PNG без альфа-канала.')
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.mode!= 'L' or image.size != (N, N) or image.n_frames !=1 or 'transparency' in image.info:
                raise InputError(prefix + 'Требуется одно непрозрачное изображение в режиме L.')
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            return np.array(image,dtype=np.uint8)
    except (OSError, SyntaxError, ValueError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(prefix+ 'PNG повреждён или не читается.') from exc
def load_targets( raw_images):
    if len(raw_images) !=4:
        raise InputError('Загрузите четыре изображения, по одному на плоскость.')
    pixels = np.stack([png_array(raw, j).astype(np.float32) for j, raw in enumerate(raw_images, 1)])
    return pixels / np.maximum(pixels.max(axis=(-2, -1 ), keepdims= True) ,1e-12)
def png_bytes(q):
    stream= io.BytesIO()
    Image.fromarray(q).save(stream, format='PNG')
    return stream.getvalue()
def gray_bytes(a):
    return png_bytes(np.rint(np.clip(a,0, 1 ) *255).astype('uint8'))
class Refiner(UpdateNet):
    """закоголок не меняет нейм чекпоинта"""
    def __init__(self):
        super().__init__(24)
        self.log_gain= nn.Parameter(torch.zeros(()))
    def forward(self, features):
        v = self.body(features).float()[:, 0]
        gain = .35 + .30 *torch.tanh(self.log_gain)
        return gain * torch.tanh(v)
class Engine:
    def __init__(self, device=None):
        self.device= str(torch.device(device or os.environ.get('DOE_DEVICE', 'cpu')))
        torch.set_num_threads(2)
        config=json.loads((WEIGHTS/ 'config.json').read_text(encoding='utf-8'))
        if (config['shape'], config['wavelength_nm'] , config['pixel_pitch_um'], config['depths_mm']) != (
            [N, N], round(WAVELENGTH * 1e9), round(PITCH* 1e6), [ round(z* 1e3, 1) for z in DEPTHS]
        ):
            raise ValueError('Физическая конфигурация не совпадает с реализацией.')
        self.config =config
        self.op= Fresnel(self.device)
        self.features = PhysicsFeatures(self.op)
        self.optics=OpticalUpdates(self.op)
        self.init = self.load('init', InitNet(True, 32))
        self.update= self.load('update', UpdateNet(24))
        self.refiner= self.load('refiner', Refiner())
    def load(self, name, model):
        record =self.config['weights'][ name]
        raw = (WEIGHTS / record['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError(f'Нарушена контрольная сумма весов {name}.')
        checkpoint= torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        if self.device !='cpu':
            model= model.to(self.device)
        return model.to(memory_format=torch.channels_last).eval().requires_grad_(False)
    @torch.no_grad()
    def synthesize(self, t):
        t = torch.from_numpy(t)[None]
        if self.device !='cpu':
            t = t.to(self.device)
        p = self.optics(self.update, t, infer(self.init, t, self.features), self.config['base_steps'] )
        for _ in range(self.config['refine_steps']):
            x= optical_features(self.optics, t, p, aligned= True)
            p = p +self.refiner(x.contiguous(memory_format=torch.channels_last))
        return png_bytes(encode(p[ 0].cpu().numpy()))
    @staticmethod
    def export(phase):
        return png_bytes(encode(phase.detach()[0].cpu().numpy()))
    def decode(self, mask):
        return torch.tensor(decode(png_array(mask)), dtype=torch.float32, device=self.device)[None]
    @torch.no_grad()
    def replay(self, mask):
        return self.op.reconstruct(self.decode(mask))[2][0].cpu().numpy()
    def refine_mask(self, target, initial, budget):
        t =torch.from_numpy(target)[None].to(self.device)
        def check(mask):
            return evaluate(target, self.replay(mask))
        def score(metrics):
            return (metrics['ssim_global_mean'], metrics['ssim_global_min'],
                    metrics['pearson_mean'], metrics[ 'pearson_min'])
        return refine(self.op, t, self.decode(initial) ,initial, budget, self.export, check, score)
_lock = Lock()
_engine = None
def predict(raw_images, diagnostics=False, *, engine=None):
    """1 канон путь"""
    target = load_targets(raw_images)
    global _engine
    with _lock:
        if engine is None:
            if _engine is None:
                _engine= Engine()
            engine = _engine
        start = perf_counter()
        initial_fft = engine.op.count + engine.optics.unit.count + engine.features.unit.count
        mask= engine.synthesize(target)
        mask, trace, _ = engine.refine_mask(target, mask, FINAL_BUDGET)
        milliseconds = (perf_counter() -start) *1000
        amplitude = engine.replay(mask) if diagnostics else None
        forward_fft =engine.op.count + engine.optics.unit.count + engine.features.unit.count - initial_fft
    if not diagnostics:
        return mask
    return mask, target, amplitude, evaluate(target, amplitude) | trace | {
        'mode': 'hybrid_quality', 'preset': FINAL_PRESET,
        'synthesis_ms': milliseconds, 'forward_fft': forward_fft,
        'fft_total': forward_fft + 4 * trace['backward_calls']}
def comparison(target, amplitude):
    tile, row = 192,226
    im = Image.new('RGB', (4 * tile, 2 * row + 32), '#eef2f6')
    draw= ImageDraw.Draw(im)
    for j, z in enumerate(DEPTHS):
        draw.text((j * tile + 6,8), f'Plane {j+1} | {z * 1e3:.1f} mm', fill='#182a3a')
    for i, (label, values) in enumerate((('Target', target), ('Reconstruction', amplitude))):
        for j, a in enumerate(values):
            q= np.rint(np.clip(a, 0, 1) *255).astype('uint8')
            draw.text((j * tile +6, 32 + row * i), label, fill='#182a3a')
            im.paste(Image.fromarray(q).resize((tile, tile), Image.Resampling.NEAREST),
                     (j *tile,56 + row *i))
    output = io.BytesIO()
    im.save(output, format ='PNG')
    return output.getvalue()
















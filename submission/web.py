import base64
import io
import json
from pathlib import Path
import zipfile
from flask import Flask, jsonify, request, send_from_directory
from .backend import MAX_PNG, InputError, comparison, gray_bytes, predict

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * MAX_PNG + 32 * 1024
EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'

@app.get('/')
def index():
    return app.send_static_file('index.html')
@app.get('/examples/<set_id>/<int:plane>')
def example(set_id, plane):
    if set_id not in ('run_01', 'run_02') or plane not in (1 ,2, 3 ,4):
        return '', 404
    return send_from_directory(EXAMPLES /set_id / 'targets', f'plane_{plane}.png')
@app.errorhandler( 413)
def too_large(_):
    return jsonify(error='Суммарный размер файлов слишком велик.'), 413
@app.post('/api/synthesize')
def synthesize():
    try:
        fields= [request.files.getlist(f'plane_{j}') for j in range(1, 5)]
        if any(len(field)!= 1 for field in fields) or len(request.files) !=4:
            raise InputError('Загрузите ровно четыре PNG, по одному на плоскость.')
        raw =[field[0].stream.read(MAX_PNG +1) for field in fields]
        mask, targets, amplitude, metrics= predict(raw, diagnostics=True)
    except InputError as exc:
        return jsonify(error=str(exc)), 400
    recon = [gray_bytes(a) for a in amplitude]
    preview= comparison(targets,amplitude)
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED ) as z:
        z.writestr('doe_phase.png' , mask)
        z.writestr('comparison.png', preview)
        z.writestr('metrics.json', json.dumps(metrics, indent=2))
        for j, (original, reconstruction) in enumerate(zip(raw, recon), 1):
            z.writestr(f'targets/plane_{ j}.png' , original)
            z.writestr(f'reconstructions/plane_{j}.png',  reconstruction)
    def b64(data):
        return base64.b64encode( data).decode('ascii' )
    return jsonify(mask= b64(mask), targets= [b64(gray_bytes(x)) for x in targets],
                   reconstructions=[b64( x) for x in recon], metrics =metrics,
                   archive=b64(bundle.getvalue() ))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port =7860, debug= False, threaded=False)









    

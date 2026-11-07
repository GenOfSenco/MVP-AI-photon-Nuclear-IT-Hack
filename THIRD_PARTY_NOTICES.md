# Сторонние компоненты и демонстрационные данные

Зависимости устанавливаются отдельно из пакетов, указанных в requirements.txt. Их исходный код в эту папку не копируется.

| Компонент | Лицензия пакета |
|---|---|
| PyTorch | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| Pillow | HPND |
| Flask | BSD-3-Clause |

Комплекты 01, 02 и 04 получены в проекте процедурно или рассчитаны физически. Комплект 03 содержит кропы изображений из набора scikit-image. Исходные сведения для каждой плоскости приведены в [документации scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html).

| Плоскость | Изображение | Автор или источник | Статус |
|---|---|---|---|
| 1 | chelsea | Stefan van der Walt | CC0 |
| 2 | clock | Stefan van der Walt | Public domain |
| 3 | astronaut | NASA | Public domain |
| 4 | rocket | SpaceX | Public domain |

Исходный код и веса самого проекта распространяются на условиях, определяемых их правообладателями.

## Демонстрация рукописных цифр

Источник: [MNIST](https://keras.io/api/datasets/mnist/), Yann LeCun и Corinna Cortes, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Четыре тестовые записи взяты из [зеркала CVDF](https://github.com/cvdfoundation/mnist). Полные изображения 28×28 увеличены до 112×112 и размещены в центре чёрного холста 128×128. [Пятая панель](assets/png/example_05.png), включающая эти адаптации, распространяется по CC BY-SA 3.0.


(указываем так как некоторые хакатоны требуют, точных требований по этому небыло, тем не менее лишним не будет)
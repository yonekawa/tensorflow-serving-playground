# tensorflow-serving-playground
A playground of [Tensorflow Serving](https://www.tensorflow.org/serving)

## Export Tensorflow models

```bash
$ pip install -r requirements.txt
$ python export/mnist_tensorflow.py --model_version=2 models/mnist
```

### (WIP)VGG - Keras

```bash
$ python export/keras_vgg.py models/vgg
```

NOTE: Waiting for a new Tensorflow release with [this problem](https://github.com/tensorflow/tensorflow/issues/14284) fixed.

## Run model server

```bash
$ docker build --pull -t tensorflow-model-server -f Dockerfile .
$ docker run -d -v $(pwd):/tmp -p 9000:9000 tensorflow-model-server tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/models/mnist
```

## Request to prefictions

You can use [myleott/mnist_png](https://github.com/myleott/mnist_png)

```bash
$ go run serving_client/mnist/client.go --serving-address localhost:9000 mnist_png/1/1039.png
```



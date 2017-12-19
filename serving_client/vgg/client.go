package main

import (
	"context"
	"flag"
	"image/jpeg"
	"log"
	"os"

	tfcoreframework "tensorflow/core/framework"
	pb "tensorflow_serving/apis"

	"fmt"

	"path/filepath"

	"encoding/json"

	"github.com/nfnt/resize"
	"google.golang.org/grpc"
)

func main() {
	servingAddress := flag.String("serving-address", "localhost:9000", "The tensorflow serving address")
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Println("Usage: " + os.Args[0] + " --serving-address localhost:9000 path/to/img.jpg")
		os.Exit(1)
	}

	imgPath, err := filepath.Abs(flag.Arg(0))
	if err != nil {
		log.Fatalln(err)
	}

	i, err := os.Open(imgPath)
	if err != nil {
		log.Fatalln(err)
	}
	defer i.Close()

	jp, err := jpeg.Decode(i)
	if err != nil {
		log.Fatalln(err)
	}
	jp = resize.Resize(224, 224, jp, resize.Lanczos3)

	floats := make([]float32, 224*224*3)
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			r, g, b, _ := jp.At(x, y).RGBA()
			ch := []uint32{r, g, b}
			for c := 0; c < 3; c++ {
				floats[c+(x*3)+(y*224*3)] = float32(ch[c] / 255)
			}
		}

	}

	request := &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name: "vgg",
		},
		Inputs: map[string]*tfcoreframework.TensorProto{
			"input_1": &tfcoreframework.TensorProto{
				Dtype: tfcoreframework.DataType_DT_FLOAT,
				TensorShape: &tfcoreframework.TensorShapeProto{
					Dim: []*tfcoreframework.TensorShapeProto_Dim{
						&tfcoreframework.TensorShapeProto_Dim{
							Size: int64(1),
						},
						&tfcoreframework.TensorShapeProto_Dim{
							Size: int64(224),
						},
						&tfcoreframework.TensorShapeProto_Dim{
							Size: int64(224),
						},
						&tfcoreframework.TensorShapeProto_Dim{
							Size: int64(3),
						},
					},
				},
				FloatVal: floats,
			},
		},
	}

	conn, err := grpc.Dial(*servingAddress, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Cannot connect to the grpc server: %v\n", err)
	}
	defer conn.Close()

	client := pb.NewPredictionServiceClient(conn)

	resp, err := client.Predict(context.Background(), request)
	if err != nil {
		log.Fatalln(err)
	}

	index := 0
	score := float32(0.0)
	for i, s := range resp.Outputs["predictions"].FloatVal {
		if score < s {
			score = s
			index = i
		}
	}

	classes := loadImageNetClasses()
	key := fmt.Sprintf("%d", index)
	fmt.Printf("%v %v proba: %v\n", classes[key][0], classes[key][1], score)
}

func loadImageNetClasses() map[string][]string {
	f, err := os.Open(filepath.Join("misc", "imagenet_class_index.json"))
	if err != nil {
		log.Fatalln(err)
	}
	defer f.Close()

	m := make(map[string][]string)
	if err := json.NewDecoder(f).Decode(&m); err != nil {
		log.Fatalln(err)
	}

	return m
}

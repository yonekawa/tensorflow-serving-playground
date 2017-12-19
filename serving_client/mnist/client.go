package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	tfcoreframework "tensorflow/core/framework"
	pb "tensorflow_serving/apis"

	"image/png"

	"google.golang.org/grpc"
)

func main() {
	servingAddress := flag.String("serving-address", "localhost:9000", "The tensorflow serving address")
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Println("Usage: " + os.Args[0] + " --serving-address localhost:9000 path/to/img.png")
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

	p, err := png.Decode(i)
	if err != nil {
		log.Fatalln(err)
	}

	floats := make([]float32, 28*28)
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			r, _, _, _ := p.At(i, j).RGBA()
			floats[i+(j*28)] = float32(r) / 255
		}
	}

	request := &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name:          "mnist",
			SignatureName: "predict_images",
		},
		Inputs: map[string]*tfcoreframework.TensorProto{
			"images": &tfcoreframework.TensorProto{
				Dtype: tfcoreframework.DataType_DT_FLOAT,
				TensorShape: &tfcoreframework.TensorShapeProto{
					Dim: []*tfcoreframework.TensorShapeProto_Dim{
						&tfcoreframework.TensorShapeProto_Dim{
							Size: int64(1),
						},
						&tfcoreframework.TensorShapeProto_Dim{
							Size: int64(28),
						},
						&tfcoreframework.TensorShapeProto_Dim{
							Size: int64(28),
						},
						&tfcoreframework.TensorShapeProto_Dim{
							Size: int64(1),
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

	for i, s := range resp.Outputs["scores"].FloatVal {
		if s == 1.0 {
			fmt.Print(i)
			break
		}
	}
}

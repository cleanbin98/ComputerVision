using System;
using System.IO;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace Project
{
    class Program
    {
        static void Main(string[] args)
        {
            const string config = "C:/ObjectInference/graph.pbtxt";
            const string model = "C:/ObjectInference/frozen_inference_graph.pb";
            string[] classNames = File.ReadAllLines("C:/ObjectInference/labelmap.txt");

            Mat image = new Mat("C:/ImageSamle/umbrella.jpg");
            Mat dst = new Mat();
            Cv2.Resize(image, dst, new Size(1080, 720));
            Net net = Net.ReadNetFromTensorflow(model, config);
            Mat inputBlob = CvDnn.BlobFromImage(dst, 1, new Size(300, 300), swapRB: true, crop: false);

            net.SetInput(inputBlob);
            Mat outputBlobs = net.Forward();

            Mat prob = new Mat(outputBlobs.Size(2), outputBlobs.Size(3), MatType.CV_32F, outputBlobs.Ptr(0));
            for (int p = 0; p < prob.Rows; p++)
            {
                float confidence = prob.At<float>(p, 2);
                if (confidence > 0.9)
                {
                    int classes = (int)prob.At<float>(p, 1);
                    string label = classNames[classes];

                    int x1 = (int)(prob.At<float>(p, 3) * dst.Width);
                    int y1 = (int)(prob.At<float>(p, 4) * dst.Height);
                    int x2 = (int)(prob.At<float>(p, 5) * dst.Width);
                    int y2 = (int)(prob.At<float>(p, 6) * dst.Height);

                    Cv2.Rectangle(dst, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 0, 255));
                    Cv2.PutText(dst, label, new Point(x1, y1), HersheyFonts.HersheyComplex, 1.0, Scalar.Red);
                }
            }
            Cv2.ImShow("image", dst);
            Cv2.WaitKey();
            Cv2.DestroyAllWindows();
        }
    }
}
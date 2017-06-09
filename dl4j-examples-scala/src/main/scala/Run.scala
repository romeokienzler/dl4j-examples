import java.util.Queue;

import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.FFTInstance;

import com.google.gson.JsonObject;
import com.ibm.iotf.client.app.Command;
import com.ibm.iotf.client.app.Event;
import com.ibm.iotf.client.app.EventCallback;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;

object Run {
  val windowSize = 3000
  def main(args: Array[String]): Unit = {
    val lstm: IoTAnomalyExampleLSTMFFTWatsonIoT = new IoTAnomalyExampleLSTMFFTWatsonIoT(windowSize * 6)
    object MyEventCallback extends EventCallback {

      var fifo: Queue[Array[Double]] = new CircularFifoQueue[Array[Double]](windowSize)
      var index = 0

      def fft(x: Array[Double]): Array[Double] = {
        var fftDo = new DoubleFFT_1D(x.length);
        var fft = new Array[Double](x.length * 2);
        System.arraycopy(x, 0, fft, 0, x.length);
        fftDo.realForwardFull(fft);
        return fft;
      }

      override def processEvent(arg0: Event) {
        val json = arg0.getData().asInstanceOf[JsonObject]
        def conv = { v: Object => v.toString.toDouble }
        val event: Array[Double] = Array(conv(json.get("x")), conv(json.get("y")), conv(json.get("z")))

        fifo.add(event)

        if ((index < windowSize - 1) && (index % 100 == 0)) {
          println("Waiting for tumbling window to fill: " + index)
        }
        if (index >= windowSize - 1) {
          val ixNd = Nd4j.create(fifo.toArray(Array.ofDim[Double](windowSize, 3)));
          def xtCol = { (x: INDArray, i: Integer) => x.getColumn(i).dup.data.asDouble }
          val fftXYZ = Nd4j.hstack(Nd4j.create(fft(xtCol(ixNd, 0))), Nd4j.create(fft(xtCol(ixNd, 1))), Nd4j.create(fft(xtCol(ixNd, 2))));

          println(lstm.detect(fftXYZ));
          fifo = new CircularFifoQueue[Array[Double]](windowSize);
          index = index - 1;
        }
        index = index + 1;
      }

      override def processCommand(arg0: Command) {
        println(arg0);
      }
    };
    new WatsonIoTConnector(MyEventCallback);
  }
}
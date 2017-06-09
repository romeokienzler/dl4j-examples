

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.datavec.api.transform.transform.normalize.Normalize;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;

/**
 * @author Romeo Kienzler (based on the MNISTAnomalyExample of Alex Black)
 */
class IoTAnomalyExampleLSTMFFTWatsonIoT(windowSize: Integer) {
  // Random number generator seed, for reproducability
  val seed = 12345
  // Network learning rate
  val learningRate = 0.01
  val rng = new Random(seed)

  // Set up network. 784 in/out (as MNIST images are 28x28).
  // 784 -> 250 -> 10 -> 250 -> 784
  val conf = new NeuralNetConfiguration.Builder()
    .seed(12345)
    .iterations(1)
    .weightInit(WeightInit.XAVIER)
    .updater(Updater.ADAGRAD)
    .activation("relu")
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(learningRate)
    .regularization(true)
    .l2(0.0001)
    .list()
    .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(windowSize).nOut(10)
      .build())
    .layer(1, new VariationalAutoencoder.Builder()
      .activation(Activation.LEAKYRELU)
      .encoderLayerSizes(256, 256) //2 encoder layers, each of size 256
      .decoderLayerSizes(256, 256) //2 decoder layers, each of size 256
      .pzxActivationFunction(Activation.IDENTITY) //p(z|data) activation function
      //Bernoulli reconstruction distribution + sigmoid activation - for modelling binary data (or data in range 0 to 1)
      .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
      .nIn(10) //Input size: 28x28
      .nOut(10) //Size of the latent variable space: p(z|x) - 32 values
      .build())
    .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
      .activation(Activation.IDENTITY).nIn(10).nOut(windowSize).build())
    .pretrain(false).backprop(true).build();

  val net = new MultiLayerNetwork(conf)

//  val tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
//    .averagingFrequency(5)
//    .workerPrefetchNumBatches(2)
//    .batchSizePerWorker(16)
//    .build();
//
//  val sparkNet = new SparkDl4jMultiLayer(new SparkContext(), conf, tm);

  net.setListeners(Collections.singletonList(new ScoreIterationListener(1).asInstanceOf[IterationListener]))

  def detect(xyz: INDArray): Double = {
    for (a <- 1 to 1000) {
      net.fit(xyz, xyz)
    }
    return net.score(new DataSet(xyz, xyz))
  }
}

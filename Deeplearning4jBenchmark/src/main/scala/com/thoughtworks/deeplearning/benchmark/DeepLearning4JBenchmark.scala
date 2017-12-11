package com.thoughtworks.deeplearning.benchmark

import com.thoughtworks.deeplearning.etl.Cifar10
import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.api.storage.{StatsStorageEvent, StatsStorageListener}
import org.deeplearning4j.nn.api.{Model, OptimizationAlgorithm}
import org.deeplearning4j.nn.conf.graph.{ElementWiseVertex, MergeVertex, ScaleVertex}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.inputs.InputType.InputTypeConvolutionalFlat
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, LossLayer}
import org.deeplearning4j.nn.conf.preprocessor.{CnnToFeedForwardPreProcessor, RnnToFeedForwardPreProcessor}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.preprocessor.ImageFlatteningDataSetPreProcessor
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * @author 杨博 (Yang Bo)
  */
object DeepLearning4JBenchmark {
  val HiddenFeatures = 256

  val conf = new NeuralNetConfiguration.Builder()
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .updater(Updater.SGD)
    .learningRate(0.0001)
    .regularization(true)
    .l2(0.000001)
    .graphBuilder()
    .addInputs("input")
    .addLayer("branch1layer1",
              new DenseLayer.Builder()
                .nIn(Cifar10.NumberOfPixelsPerSample)
                .nOut(HiddenFeatures)
                .activation(Activation.RELU)
                .build(),
              "input")
    .addLayer("branch1layer2",
              new DenseLayer.Builder()
                .nIn(HiddenFeatures)
                .nOut(Cifar10.NumberOfClasses)
                .activation(Activation.RELU)
                .build(),
              "branch1layer1")
    .addLayer("branch2layer1",
              new DenseLayer.Builder()
                .nIn(Cifar10.NumberOfPixelsPerSample)
                .nOut(HiddenFeatures)
                .activation(Activation.RELU)
                .build(),
              "input")
    .addLayer("branch2layer2",
              new DenseLayer.Builder()
                .nIn(HiddenFeatures)
                .nOut(Cifar10.NumberOfClasses)
                .activation(Activation.RELU)
                .build(),
              "branch2layer1")
    .addLayer("branch3layer1",
              new DenseLayer.Builder()
                .nIn(Cifar10.NumberOfPixelsPerSample)
                .nOut(HiddenFeatures)
                .activation(Activation.RELU)
                .build(),
              "input")
    .addLayer("branch3layer2",
              new DenseLayer.Builder()
                .nIn(HiddenFeatures)
                .nOut(Cifar10.NumberOfClasses)
                .activation(Activation.RELU)
                .build(),
              "branch3layer1")
    .addLayer("branch4layer1",
              new DenseLayer.Builder()
                .nIn(Cifar10.NumberOfPixelsPerSample)
                .nOut(HiddenFeatures)
                .activation(Activation.RELU)
                .build(),
              "input")
    .addLayer("branch4layer2",
              new DenseLayer.Builder()
                .nIn(HiddenFeatures)
                .nOut(Cifar10.NumberOfClasses)
                .activation(Activation.RELU)
                .build(),
              "branch4layer1")
    .addVertex("add",
               new ElementWiseVertex(ElementWiseVertex.Op.Add),
               "branch1layer2",
               "branch2layer2",
               "branch3layer2",
               "branch4layer2")
    .addVertex("scale", new ScaleVertex(0.25), "add")
    .addLayer("out",
              new LossLayer.Builder()
                .activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .build(),
              "scale")
    .setOutputs("out")
    .backprop(true)
    .build()

  def main(args: Array[String]): Unit = {
    val model = new ComputationGraph(conf)
    model.init()

    model.setListeners(new PerformanceListener(1))

    import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
    val cifar = new CifarDataSetIterator(64, CifarLoader.NUM_TRAIN_IMAGES)
    cifar.setPreProcessor(new ImageFlatteningDataSetPreProcessor)
    model.fit(cifar)

  }

}

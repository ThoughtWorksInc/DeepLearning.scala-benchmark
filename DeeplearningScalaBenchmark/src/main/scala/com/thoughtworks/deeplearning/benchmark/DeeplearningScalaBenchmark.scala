package com.thoughtworks.deeplearning.benchmark

import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deeplearning.etl.Cifar10
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.plugins._
import com.thoughtworks.feature.Factory
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.future._
import scala.concurrent.Await
import scala.concurrent.duration.Duration
import com.thoughtworks.each.Monadic._
import scalaz.std.iterable._
import scalaz.std.stream._
import scalaz.syntax.all._

import java.util.concurrent.Executors
import scala.concurrent.ExecutionContext

object DeeplearningScalaBenchmark {
  import $exec.`https://gist.github.com/Atry/1fb0608c655e3233e68b27ba99515f16/raw/39ba06ee597839d618f2fcfe9526744c60f2f70a/FixedLearningRate.sc`
  import $exec.`https://gist.github.com/Atry/8a52557ea68b269d74d0a09c4282a267/raw/4624d8612e189d5281a2fb82312162d485cefd8c/L2Regularization.sc`
  import $exec.`https://gist.github.com/Rabenda/0c2fc6ba4cfa536e4788112a94200b50/raw/233cbc83932dad659519c80717d145a3983f57e1/Adam.sc`
//  val singleThreadExecutor = Executors.newSingleThreadExecutor()
//  implicit val singleThreadExecutionContext = ExecutionContext.fromExecutor(singleThreadExecutor)
  import scala.concurrent.ExecutionContext.Implicits._

  val hyperparameters = Factory[Builtins with L2Regularization with Adam with FixedLearningRate].newInstance(
    learningRate = 0.0001,
    l2Regularization = 0.000001
  )
  import hyperparameters._, implicits._

  def softmax(scores: INDArrayLayer): INDArrayLayer = {
    val expScores = exp(scores)
    expScores / expScores.sum(1)
  }

  def logLoss(probabilities: INDArrayLayer, expectOutput: INDArray): DoubleLayer = {
    -(hyperparameters.log(probabilities) * expectOutput).mean
  }

  val HiddenFeatures = 256

  final case class Branch(
      affineWeight0: INDArrayWeight,
      affineBias0: INDArrayWeight,
      affineWeight1: INDArrayWeight,
      affineBias1: INDArrayWeight
  ) extends (INDArray => INDArrayLayer) {
    def apply(input: INDArray) = {
      val layer0 = max(input dot affineWeight0 + affineBias0, 0.0)
      val layer1 = max(layer0 dot affineWeight1 + affineBias1, 0.0)
      layer1
    }

    def this() = {
      this(
        affineWeight0 = INDArrayWeight(
          Nd4j
            .randn(Cifar10.NumberOfPixelsPerSample, HiddenFeatures)
            .div(math.sqrt(Cifar10.NumberOfPixelsPerSample / 2))),
        affineBias0 = INDArrayWeight(Nd4j.randn(1, HiddenFeatures).mul(1e-5)),
        affineWeight1 =
          INDArrayWeight(Nd4j.randn(HiddenFeatures, Cifar10.NumberOfClasses).div(math.sqrt(HiddenFeatures / 2))),
        affineBias1 = INDArrayWeight(Nd4j.randn(1, Cifar10.NumberOfClasses).mul(1e-5))
      )
    }
  }

  final case class FusionModel(branches: Branch*) extends (INDArray => INDArrayLayer) {
    def apply(input: INDArray) = {
      branches.view.map(_(input)).reduce(_ + _) * (1.0 / branches.length)
    }
    def this() = this(Seq.fill(4)(new Branch()): _*)
  }

  private def array4DTo2D(array4d: INDArray): INDArray = {
    val Array(numberOfSamples, width, height, depth) = array4d.shape()
    array4d.reshape(numberOfSamples, width * height * depth)
  }

  def main(arguments: Array[String]): Unit = {

    Cifar10
      .load()
      .flatMap { cifar10 =>
        val model = new FusionModel()
        val epoch = new Iterable[Cifar10.Batch] { def iterator = cifar10.epoch(64) }
        epoch.zipWithIndex.traverseU_ {
          case (batch, i) =>
            println(i)
            logLoss(softmax(model(array4DTo2D(batch.pixels))), batch.labels).train
        }
      }
      .blockingAwait

  }
}

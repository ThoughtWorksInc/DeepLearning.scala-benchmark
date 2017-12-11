libraryDependencies += "org.datavec" % "datavec-data-image" % "0.8.0"

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.8.0"

// The native backend for nd4j.
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.8.0" % Runtime

libraryDependencies += "org.slf4j" % "slf4j-jdk14" % "1.7.25" % Runtime

libraryDependencies += "com.thoughtworks.deeplearning.etl" %% "cifar10" % "1.1.0"

scalaVersion := "2.11.12"

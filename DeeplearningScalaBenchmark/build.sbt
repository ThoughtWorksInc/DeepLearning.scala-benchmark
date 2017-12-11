libraryDependencies += "com.thoughtworks.deeplearning.etl" %% "cifar10" % "1.1.0"

// All DeepLearning.scala built-in plugins.
libraryDependencies += "com.thoughtworks.deeplearning" %% "plugins-builtins" % "2.0.3-SNAPSHOT"

// The native backend for nd4j.
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.8.0" % Runtime

libraryDependencies += "org.slf4j" % "slf4j-jdk14" % "1.7.25" % Runtime

// Uncomment the following line to switch to the CUDA backend for nd4j.
// libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0-platform" % "0.8.0"

// The magic import compiler plugin, which may be used to import DeepLearning.scala distributed in source format.
addCompilerPlugin("com.thoughtworks.import" %% "import" % "2.0.2")

// The ThoughtWorks Each library, which provides the `monadic`/`each` syntax.
libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1"

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

fork := true

scalaVersion := "2.11.12"

//scalacOptions += "-Xexperimental"
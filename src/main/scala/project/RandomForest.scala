package project

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.mllib.util.MLUtils

/**
  * Created by vyshaalnarayanam on 12/1/17.
  */
object RandomForest {

  def main(args: Array[String]):Unit = {

    val conf = new SparkConf()
    val spark = SparkSession.builder.appName("Classifier").getOrCreate()
    import spark.sqlContext.implicits._
    val training_rdd = spark.sparkContext.textFile(args(0)+"/training_data/")
    val testing_rdd = spark.sparkContext.textFile(args(0)+"/testing_data/")

    val training_df = training_rdd.map { line =>
      val parts = line.toString().split(",")
      val features = line.substring(0,line.length()-2)
      val label = parts.last.toDouble
      (label, Vectors.dense(features.split(",").map(_.toDouble)))
    }.toDF("label", "features").cache()

    val size = training_df.count()
    val negativeCount = training_df.filter(training_df("label")===0).count()
    val balancingRatio = 1 - (negativeCount*1.0/size)

    val splits = training_df.filter(training_df("label")===0).sample(true,balancingRatio*10)
      .union(training_df.filter(training_df("label")===1)).randomSplit(Array(0.7, 0.3))

    val (trainingData, validatingData) = (splits(0), splits(1))

    trainingData.cache()
    validatingData.cache()
//    val tdc = trainingData.count()
//    val vdc = validatingData.count()
//    trainingData.show()

    val testing_df = testing_rdd.map { line =>
      val parts = line.toString().split(",")
      val features = line.substring(0,line.length()-2)
      val label = parts.last
      (label, Vectors.dense(features.split(",").map(_.toDouble)))
    }.toDF("label", "features").cache()

    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(training_df)
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(training_df)

    val treeCount = 10
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(treeCount)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(validatingData)

//    // Select example rows to display.
//    predictions.select("predictedLabel", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy = " + accuracy)

    val lp = predictions.select( "label", "prediction").cache()

    val counttotal = predictions.count()
    val correct = lp.filter($"label" === $"prediction").count()
    val wrong = lp.filter(!($"label" === $"prediction")).count()
    val trueN = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count()
    val trueP = lp.filter($"prediction" === 1.0).filter($"label" === $"prediction").count()
    val falseN = lp.filter($"prediction" === 0.0).filter(!($"label" === $"prediction")).count()
    val falseP = lp.filter($"prediction" === 1.0).filter(!($"label" === $"prediction")).count()
    val ratioWrong=wrong.toDouble/counttotal.toDouble
    val ratioCorrect=correct.toDouble/counttotal.toDouble

    val predictions1 = model.transform(testing_df)

    predictions1.select("prediction").write.csv(args(1)+"/labels")
    var stats = spark.sparkContext.parallelize(Seq(treeCount,balancingRatio,accuracy))
    stats.saveAsTextFile(args(1)+"/stats")

  }
}
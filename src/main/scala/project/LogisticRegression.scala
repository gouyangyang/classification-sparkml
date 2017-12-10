package project

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.linalg.{Vector, Vectors}


/**
  * Created by vyshaalnarayanam on 12/1/17.
  */
object LogisticRegression {

  def main(args: Array[String]):Unit = {

    val conf = new SparkConf()
    val spark = SparkSession.builder.appName("LogisticRegression").config(conf).getOrCreate()

    import spark.sqlContext.implicits._
    val training_rdd = spark.sparkContext.textFile(args(0)+"/training_data/")
    val testing_rdd = spark.sparkContext.textFile(args(0)+"/testing_data/")

    val training_df = training_rdd.map { line =>
      val parts = line.toString().split(",")
      val features = line.substring(0,line.length()-2)
      val label = parts.last.toDouble
      (label, Vectors.dense(features.split(",").map(_.toDouble)))
    }.toDF("label", "features").cache()

    val splits = training_df.randomSplit(Array(0.7, 0.3))
    val (trainingData, validatingData) = (splits(0), splits(1))

    val testing_df = testing_rdd.map { line =>
      val parts = line.toString().split(",")
      val features = line.substring(0,line.length()-2)
      val label = parts.last
      (label, Vectors.dense(features.split(",").map(_.toDouble)))
    }.toDF("label", "features").cache()

    val lr = new LogisticRegression()
    lr.setMaxIter(10).setElasticNetParam(0.1)

    val model = lr.fit(trainingData)

    val predictions = model.transform(validatingData)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)

    println("Accuracy = " + accuracy)

    val predictions1 = model.transform(testing_df)
    predictions1.select("prediction").write.csv(args(1)+"/labels")
    var stats = spark.sparkContext.parallelize(Seq(accuracy))
    stats.saveAsTextFile(args(1)+"/stats")
    spark.stop()
  }
}

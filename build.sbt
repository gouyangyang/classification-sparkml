name := "Project"

version := "1.0"

scalaVersion := "2.11.12"

libraryDependencies ++= {
  val sparkVersion = "2.2.0"
  Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion
  )
}
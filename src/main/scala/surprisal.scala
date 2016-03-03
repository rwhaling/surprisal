package net.spantree.surprisal
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Row
import org.apache.spark.mllib.rdd.RDDFunctions._

object SurprisalApp {

  def main(args: Array[String]) = {
    val sc = new SparkContext("local[4]","surprisal")
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val brownPath = "/Users/rwhaling/Downloads/brown/c*"
    val lnOf2 = scala.math.log(2)

    time()
    // Parse some text files
    val allFiles = sc.wholeTextFiles(brownPath)
    // Split on line and tokenize
    val allSplit = allFiles.flatMap( {
    	case (name, content) => {
    		val shortname = name.split("/").last
    		val words = content.split("\\s+")
    		for (word <- words)
    			yield word
    	}
    })
    // Split out words from POS tags
    val allWords = allSplit.map( wt => wt.split("/") match {
        case Array(word,tag) => word
        case _ => wt
    }).toDF("word")
    allWords.show
    val total = allWords.count
    time("parsing and tokenization")

    // Group by distinct word, and count
    time()
    val counts = allWords.toDF("word").groupBy("word").count
    time("counting words")
    counts.show

    // Count of distinct words
    val distinct = counts.count
    // Calculate probability and plog
    time()
    val wordPlog = counts.map( r => {
        val word = r.getString(0)
        val count = r.getLong(1)
        val prob = count.toDouble / total
        val plog = -1 * (scala.math.log(prob) / lnOf2)
        (word,count,prob,plog)
    }).toDF("word","count","prob","plog")
    wordPlog.count
    time("calculating probability and surprisal")
    // Sort and display
    wordPlog.sort($"count".desc).show

    val text = "the cultural and ideological fissures opening in the party could take a generation to patch"
    val tokens = text.trim.split("\\W+")
    val input = sc.parallelize(tokens).map( t => (t,"input") )

    time()
    val lookupRdd = wordPlog.select("word","plog").map( r => {
        val (word,plog) = (r.getString(0),r.getDouble(1))
        (word,plog)
    })
    val joined = input.leftOuterJoin(lookupRdd)
    joined.collect.foreach(println _)
    time("join-based scoring")
    println()

    time()
    val bcLookupMap = sc.broadcast(wordPlog.select("word","plog").map( r => {
        (r.getString(0),r.getDouble(1))
    }).collectAsMap)
    val scored = sc.parallelize(tokens).map( t => (t,bcLookupMap.value.getOrElse(t,0.0))).collect.foreach(println _)
    time("broacast map-based scoring")
    sc.stop()
  }

  def foobar() = {}
}

object time {
  var lastTime = System.currentTimeMillis

  def apply():String = {
    val now = System.currentTimeMillis
    val elapsed = now - lastTime
    lastTime = now
    elapsed.toString
  }

  def apply(task:String):Unit = {
    println(task + " took " + time() + " ms")
  }
}

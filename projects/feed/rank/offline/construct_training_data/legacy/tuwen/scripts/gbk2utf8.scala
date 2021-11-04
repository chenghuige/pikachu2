package com.appfeed.sogo

import com.alibaba.fastjson.JSON
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapred.TextInputFormat
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ListBuffer

/**
  * Created by victor on 2018/11/22.
  */
class Get_base_data {

}
object Get_base_data{
  def getgbk_Text(sc: SparkContext, inPath: String,column_check:Int, minSize: Int = 32): RDD[String] = {
    val hadoopConf = sc.hadoopConfiguration
    val fs = new Path(inPath).getFileSystem(hadoopConf)
    //val len = fs.getContentSummary(new Path(inPath)).getLength / (1024 * 1024) //以MB为单位的数据大小
    //val minPart = (len / minSize).toInt //按minSize的分块数
    val gbk_input=sc.hadoopFile(inPath,classOf[TextInputFormat],classOf[LongWritable],classOf[Text],1)
      .map(p => new String(p._2.getBytes, 0, p._2.getLength, "GBK"))
    val TT_content=gbk_input.repartition(500).filter(_.split("\t").length>column_check).filter(_.split("\t")(6).contains("topic"))
    TT_content
  }
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setAppName("dssm_samples_prepare")
    val sc = new SparkContext(sparkConf)
    val input = getgbk_Text(sc, args(0), 9).map(f => {
      val mid = f.split("\t")(0)
      val doc_id = f.split("\t")(1)
      val doc_info = f.split("\t")(6)
      val json = JSON.parseObject(doc_info)
      val app = f.split("\t")(9)
      //val video_sig = json.getOrDefault("video_sig", "0").toString.toInt
      /*
      val video_sig = if(json.get("video_sig") != null){
        json.get("video_sig").toString.toInt
      } else{
        1
      }
      */
      val video_sig = if(doc_info.contains("video_sig")){
        json.get("video_sig").toString.toInt
      } else {
        1
      }
      val title = if(doc_info.contains("title")){
        json.get("title").toString
      } else{
        "0"
      }


      var str = new ListBuffer[String]
      if(doc_info.contains("keywords_content")){
        var kw_c = json.getJSONArray("keywords_content")
        for(w <- 0 to (kw_c.size()-1)){
          str.append(kw_c.get(w).toString)
        }
      } else {
        str.append("-1")
      }

      if(doc_info.contains("keywords_secondary")){
        var kw_s = json.getJSONArray("keywords_secondary")
        for(w <- 0 to (kw_s.size()-1)){
          str.append(kw_s.get(w).toString)
        }
      } else {
        str.append("-1")
      }



      val click = f.split("\t")(8).toInt
      //val title = json.get("title").toString


      (mid, doc_id, click, video_sig, title, app, str.mkString(" "))
    }).filter(_._5 != "0").filter(_._6 == "sgsapp").map(f => f._1 + "\t" + f._2 + "\t" +  f._5 + "\t" + f._3 + "\t" + f._7)
    input.saveAsTextFile(args(1))
  }

}

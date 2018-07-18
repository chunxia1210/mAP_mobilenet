
 mAP计算过程

　　开源项目来源ｇｉｔｈｕｂ地址：https://github.com/Cartucho/mAP

step 1:

　　git clone https://github.com/Cartucho/mAP


step 2:数据准备

　 （1）生成ground-truth的txt格式文件
 
   txt文件格式 ：
   
   <class_name> <left> <top> <right> <bottom>

  
   mAP/extra/下有写好的数据格式转换脚本,如xml文件转换成需要的txt文件，调用get_displayname（）可切换<class_name>为ｄｉsplayname.
   
   第一步：将要转换的xml文件保存到mAP/ground-truth/
   第二步：python convert_gt_xml.py
  
  
  (2)生成predicted的txt格式文件

   第一步：将要测试的图片存到images/
   
   python resnet_ssd_detect.py    ### resnet网络测试

　ｏｒ　　

　　python mobilenet_ssd_detect.py     ##### mobilenet网络测试
   
   注：测试结果以.ｔｘｔ文件形式保存至指定目录

 　（3）copy（2）生成的txt文件至mAP/predicted/

step 3 :
   
   python main.py
    
   注：predicted/　与　ｇｒound-truth/中ｔｘｔ文件数目需保持一致
    
    

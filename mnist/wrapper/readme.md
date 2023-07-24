# 创建一个手写字体识别的AI能力

- **1. AI能力创建**
  - 1.1 新建AI能力
  - 1.2 填写基本信息
  - ![Image text](https://github.com/Jonyzqw/aiges_demo/blob/main/mnist/figure/基本信息.png)
  - 1.3 接口定义
    - 1.3.1 选择自定义协议
    - 1.3.2 接口类型选择“非流式Http1.1”
    - ![Image text](https://github.com/Jonyzqw/aiges_demo/blob/main/mnist/figure/接口定义.png)
    - 1.3.3 添加自定义参数（手写字体识别的数据类型int、约束类型length、取值（1-10））
    - ![Image text](https://github.com/Jonyzqw/aiges_demo/blob/main/mnist/figure/功能定义.png)
    - 1.3.4 请求数据
      - 数据类型：图片
      - 数据段名称：img
      - ![Image text](https://github.com/Jonyzqw/aiges_demo/blob/main/mnist/figure/请求数据.png)
    - 1.3.5 响应数据
      - 数据类型：文本
      - 数据段名称：文本
      - Fomat文本格式修改为plain
      - ![Image text](https://github.com/Jonyzqw/aiges_demo/blob/main/mnist/figure/响应数据.png)
  - 1.4 创建完成
  ![Image text](https://github.com/Jonyzqw/aiges_demo/blob/main/mnist/figure/创建完成.png)
- **2. 版本管理**

  - 2.1 填写版本基本信息

    - 能力版本 1.0.0
    - 版本创建方式：平台创建
    - 是否集成二郎神：否
    - ![Image text](https://github.com/Jonyzqw/aiges_demo/blob/main/mnist/figure/基本信息1.png)

  - 2.2 接口定义（检查AI能力部分是否正确）

  - 2.3 测试用例（选中基础测试用例，保存，下一步）

  - 2.4 资源上传

    - 插件实现语言：Python

    - 解码器上传：

      ```bash
      apt install s3cmd
      ```

      - 执行 $ s3cmd --configure生成配置文件，一路Enter，注意跳过认证并保存配置

        ```bash
        ......
        ...
        Test access with supplied credentials? [Y/n] n
        
        Save settings? [y/N] y
        Configuration saved to '/root/.s3cfg'
        ```

        

      - 修改.s3cfg中以下几项，也可只保留以下几项

        ```bash
        [default]
        access_key =           
        secret_key = 
        host_base = oss.xfyun.cn
        host_bucket = %(bucket)oss.xfyun.cn
        use_https = False
        #Access Key // 访问存储空间的账号ID，与引擎托管平台账号绑定，可在控制台-资源管理-密钥管理页面查看
        #Secret Key // 访问存储空间的私钥密码，与引擎托管平台账号绑定，可在控制台-资源管理-密钥管理页面查看
        #host_base //连接存储的域名或IP地址，可在引擎托管平台控制台-密钥管理页面查看，域名： oss.xfyun.cn
        ```

        

      - 列举所有 Buckets

        ```bash
        s3cmd ls --signature-v2
        ```

      - 打包引擎资源

        ```bash
        # 进入wrapper目录
        tar zcvf pywrapper.tar.gz *
        ```

      - 上传引擎资源和模型到某个 bucket

        ```bash
        s3cmd put dev.tar.gz s3://my-bucket-name/dev.tar.gz --signature-v2
        ```

        **注意：解码器云存储地址格式化为： http://my-bucket-name.oss.xfyun.cn/dev.tar.gz 格式**

      - 通过MD5检查传输数据的正确性

        ```bash
        /home/aiges/mnist/wrapper# md5sum /home/aiges/mnist/wrapper/pywrapper.tar.gz 
        2f414a75fb81fee042412e7e49b5e9c4 /home/aiges/mnist/wrapper/pywrapper.tar.gz
        ```
        ![Image text](https://github.com/Jonyzqw/aiges_demo/blob/main/mnist/figure/资源上传.png)
  - 2.5 部署规格

    - 资源类型、cpu规格、内存大小按需选择
    - 镜像来源：平台构建
    - 选择基础镜像：选择最新的aiges镜像
    - pip依赖：wrapper文件中的requirement.txt内的依赖

  - 2.6 相关配置 

    - 配置参数：

      ```bash
      key:log.dir
      value:/log/server
      ```

    - 最高并发数 and 最优并发数：1
    ![Image text](https://github.com/Jonyzqw/aiges_demo/blob/main/mnist/figure/相关配置.png)
  - 2.7 完成 -> 提交验证

**AI能力部署成功如下图所示**

​		![Image text](https://github.com/Jonyzqw/aiges_demo/blob/main/mnist/figure/能力创建成功.png)

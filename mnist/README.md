# mnist

Mnist示例
## 修改mnist/wrapper/wrapper.py文件
```javascript
{
    ctrl = StringParamField(key="ctrl", value="helloworld")
	修改为
    ctrl = StringParamField(key="ctrl", value=b"helloworld")
}
```
## 训练

训练代码来源于: [Mnist Code](https://github.com/pytorch/examples/blob/main/mnist/main.py)

稍作修改，利用CPU训练得到模型文件


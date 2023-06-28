# aiges_demo
demo wrappers for aiges

## 修改mnist/wrapper/wrapper.py文件
```javascript
{
    ctrl = StringParamField(key="ctrl", value="helloworld")
	修改为
	ctrl = StringParamField(key="ctrl", value=b"helloworld")
}
```


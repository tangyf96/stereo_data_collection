### Dependencies: 
1. MynEye SDK: https://mynt-eye-s-sdk.readthedocs.io/zh_CN/latest
2. OpenCV 3.4.1

### Method to record video:
1. Change the file name in get_video.cpp
```cpp
std::string file_name1 = "ref.avi";
std::string file_name2 = "left_trans.avi";
```
2. Build the program
```cpp
mkdir build
cd build
make
```
3. Record video
```cpp
cd build
./CollectData
```
- Enter 0
- Enter **s** to start recording the video
- Enter ctrl+c to end.

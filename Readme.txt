This is a quick guide to run the blemish app.


To run the submission file and the app successfully, follow these ease steps:

1: Unzip the file
2: Open cmd line in the same directory of the files
3: Create a build folder
4: Type ' cmake -G "Visual Studio 16 2019" ..
You can choose your own VS version in my case it was:  cmake -G "Visual Studio 17 2022" .. ' in the cmd line.
5: Inside the build folder type ' cmake --build . --config Release ' in cmd line.
6: Now, to run the executable: ..\build\Release\submission.exe


Cheers!

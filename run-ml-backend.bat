
call "C:\Users\Machine Learning\anaconda3\Scripts\activate.bat" "C:\Users\Machine Learning\anaconda3"
call conda activate label-backend

label-studio-ml start HerdNet/my_ml_backend -p 9090

:: deactivate
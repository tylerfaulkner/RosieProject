Author: Tyler Faulkner

Currently Working Using Python Version 3.6

Creating Virtual Enviroment:

	1.	Create a python virtual environment in project directory
	python -m venv kivy_tensorflow_venv

	2.	Activate the python virtual enviroment in powershell
	.\kivy_tensorflow_venv\scripts\activate

		a.	Powershell should now have (kivy_tensorflow_venv) before the path in powershell
		(kivy_tensorflow_venv) PS C:\Users\faulknert\rosieproject\kivy>

	3.	Install all dependencies from the requirements.txt
	Python -m pip install -r requirements.txt

	4. Run test_import.py to ensure everyhting installed correctly
	python test_import.py

Using Virtual Enviroment in Pycharm:

	1. Open kivy project in pycharm
	
	2. Go to the python interpreter settings
	File>Settings>Project:kivy>Python Interpreter
	
	3. Add a new Python Interpeter
	Gear Icon next to current interpreter>Add...
	
	4. Select "Existing Enviroment"
	
	5. Click on the "..." icon
		a. Select the path to the python.exe in the virtual enviroment scripts folder
			i. ..\kivy_tensorflow_venv\scripts\python.exe
	
	7. Select Ok then Ok again on the next screen
	
	8. Wait for the interpeter to be fully loaded and kivy and tensorflow python scripts should run in PyCharm
	

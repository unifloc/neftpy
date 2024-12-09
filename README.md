# neftpy

Инженерные расчеты для систем нефтяных скважин. Основан на проекте unifloc_vba (унифлок на python).

---

# Текущее состояние проекта

Должны работать расчеты PVT свойств пластовых флюидов на наборе корреляций на основе Стендинга.

папки

- docs  - сборка документации 
- neftpy  - исходный код модулей neftpy
- notebooks - примеры работы с neftpy
- sandbox - примеры и расчетные файлы для экспериментов и тестов. 

# Установка

Проект в начальной стадии разработки.
Для установки можно использовать pip 
```
$ pip install neftpy
```
В pypi версия отстает от github репозитория. Для работы с исходным кодом github рекомендуется клонировать репозиторий на локальный компьютер, создать окружение для работы с проектом и установить `neftpy` в окружении используя команду 
```
$ pip install -e .
```
Здесь точка означает "установить пакет из текущей папки в редактируемом режиме режиме"


# tips

для работы внутри проекта оказалось удобным to install the project in editable mode. 

This way, all files will be able to locate each other, always starting from your project root directory. In order to do this, follow these steps:

1) write a setup.py file and add it to your project root folder - it doesn't need much info at all:
```
# setup.py
from setuptools import setup, find_packages

setup(name='MyPackageName', version='1.0.0', packages=find_packages())
```

2) install your package in editable mode (ideally from a virtual environment). From a terminal in your project folder, write
```
$ pip install -e .
```
Note the dot - this means "install the package from the current directory in editable mode".

3) your files inside the project are now able to locate each other, always starting from the project root. To import Objective, for example, you write:
```
from mod.mods import Objective
```
This will be true to import Objective for any file, no matter where it is located in the project structure.

you should use a virtual environment for this, so that pip does not install your package to your main Python installation (which could be messy if your project has many dependencies).


https://stackoverflow.com/questions/56806620/how-to-import-modules-from-parent-sub-directory

---
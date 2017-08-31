#!/bin/bash
python2-pyuic4 main.ui > mainui.py
sed 's/QtGui.KLed/KLed/g' mainui.py > mainui2.py

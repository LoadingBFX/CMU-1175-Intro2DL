#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/27/2024 10:31 PM
# @Author  : Loading
# pip install --upgrade --force-reinstall --no-deps kaggle==1.5.8 -q
# mkdir /root/.kaggle

with open("/root/.kaggle/kaggle.json", "w+") as f:
    f.write('{"username":"","key":""}')

# chmod 600 /root/.kaggle/kaggle.json

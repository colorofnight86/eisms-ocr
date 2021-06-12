#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request
from werkzeug.utils import secure_filename
from ReceiptAutoInfoExtract import imageOcr
import json

app = Flask(__name__)


@app.route('/')
def hello_world():
    return '<p>connection is ok.</p></p>ocr识别接口：/upload</p>'


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    result = False
    try:
        if request.method == "POST":
            f = request.files['files']
            filename = './uploads/'+secure_filename(f.filename)
            f.save(filename)
            result = imageOcr(filename)

            passed = check(result)

    except Exception as ex:
        print("出现如下异常%s" % ex)
        return json.dumps({"code": 500, "msg": "Exception occurred"})
    else:
        if passed > 12:
            return json.dumps({"code": 200, "passed": str(passed)+"/20", "msg": "success",
                               "data": {"extractorResult": [result]}})
        else:
            return json.dumps({"code": 401, "passed": str(passed)+"/20", "msg": "fail",
                               "data": {"extractorResult": "Low recognition rate"}})


# 识别率指标
def check(receipt):
    passed = 0
    count = 0
    if len(receipt['invoiceCode']) == 12:
        passed += 2
    if len(receipt['invoiceNumber']) == 8:
        passed += 2
    if len(receipt['checkCode']) == 15:
        passed += 1
    if receipt['purchaserName'] != '':
        passed += 1
    if receipt['sellerName'] != '':
        passed += 1
    if receipt['purchaserName'] != '':
        passed += 1
    if receipt['sellerCode'] != '':
        passed += 1
    if receipt['sellerAddrTel'] != '':
        passed += 1
    if receipt['sellerBankCode'] != '':
        passed += 1
    if receipt['totalExpense'] != '':
        passed += 1
        count += 1
    if receipt['totalTax'] != '':
        passed += 1
        count += 1
    if receipt['totalTaxExpenseZh'] != '':
        passed += 1
    if receipt['totalTaxExpense'] != '':
        passed += 1
        count += 1
    if count > 2 and receipt['totalExpense']+receipt['totalTax'] == receipt['totalTaxExpense']:
        count += 5

    return passed


if __name__ == "__main__":
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port='5000', debug=True)

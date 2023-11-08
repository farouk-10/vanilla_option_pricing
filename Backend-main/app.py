import io
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
from flask_cors import CORS
import base64
from Classes import *
from euoption import *
from amoption import *
from getImplied import *
from plotgreeks import *
import json

app = Flask(__name__)
CORS(app)


@app.route('/calculate', methods=['POST'])
def calculate():

    #getting the data
    data= request.get_json() 
    K=float(data["strike"])
    R=float(data["interest"])/100
    ticker = data["ticker"]
    dividend = float(data["dividend"])/100
    sigma = float(data["volatility"])/100
    if len(data["period"])>0 :
        N= int(data["period"])
    else :
        N= 0
    
    
    # Save the figure to a BytesIO object.
    image_stream = io.BytesIO()
    #plt.savefig(image_stream, format='png')
    
     # Move the file pointer to the beginning of the stream.
    [delta,gamma,vega,rho,theta] = np.zeros(5)

    encoded_image=""
    if data["optionType"]=="European" :
        impliedVolatility= getReal(ticker,data["maturity"],data["option"],K)
        EuroOption = eOp(K=K,ticker=ticker,N=N,ot=data["option"],exp=data["maturity"],d=dividend,r=R,s=sigma)
        EuroOption.D()
        EuroOption.volatility()
        EuroOption.dividend()
        d = EuroOption.d
        St = float(EuroOption.df.iloc[-1,0])
        volatility = round(EuroOption.s*100,3)
        if data["model"]== "Black & Scholes" :
            price = EuroOption.BS()
            delta = EuroOption.delta()
            gamma = EuroOption.gamma()
            vega = EuroOption.vega()
            rho = EuroOption.rho()
            theta = EuroOption.theta()
        elif data["model"]== "Binomial" :
            price = EuroOption.CRR()
            EuroOption.P_CRR(image_stream)
            image_stream.seek(0)
            encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
        else :
            price = EuroOption.TM()
            EuroOption.P_TM(image_stream)
            image_stream.seek(0)
            encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
    elif data["optionType"] == "American" :
        impliedVolatility= ["0","0"]
        UsOption = aOp(K=K,ticker=ticker,N=N,ot=data["option"],exp=data["maturity"],d=dividend, r=R,s=sigma)
        UsOption.D()
        UsOption.volatility()
        UsOption.dividend()
        d = UsOption.d
        St = float(UsOption.df.iloc[-1,0])
        volatility = round(UsOption.s*100,3)
        if data["model"] == "Binomial" : 
            price = UsOption.CRR()
            UsOption.P_CRR(image_stream)
            image_stream.seek(0)
            encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
        elif data["model"] =="Trinomial" :
            price = UsOption.TM()
            UsOption.P_TM(image_stream)
            image_stream.seek(0)
            encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
        
    
    return jsonify({'price': round(price,3),"image": encoded_image, "st" :round(St,3), "volatility" :volatility,"impliedVolatility": impliedVolatility[1],"realPrice": impliedVolatility[0], "dividendYield" : round(d*100,3),"delta" :round(delta,2),"gamma" : round(gamma,2),"vega" :round(vega,2),"rho" :round(rho,2), "theta" : round(theta,2)  })
    

@app.route('/getDates', methods=['POST'])
def getDates() :
    data= request.get_json()

    expiries = Op.expiries(data["ticker"])

    return jsonify({"expiries" : expiries})

@app.route('/getPlot', methods=['POST'])
def getPlot() : 
    data= request.get_json()
    x=[]
    y=[]
    plotOption= P_eOp(K=float(data["strike"]),ticker=data["ticker"],N=0,ot=data["ot"],exp=data["exp"],r=float(data["r"]),d=float(data["d"]),s=float(data["s"]))
    plotOption.D()
    if data["z"]=="None" :
        x,Y=plotOption.P2_OP(float(data["xmin"]),float(data["xmax"]),data["x"])
        match data["y"] :
            case "BS" :
                y=Y[0].tolist()
            case "delta" :
                y=Y[1].tolist()
            case "gamma" :
                y=Y[2].tolist()
            case "rho" :
                y=Y[3].tolist()
            case "theta" :
                y=Y[4].tolist()
            case "vega" :
                y=Y[5].tolist()
        return jsonify({"x" :x.tolist() , "y" : y})
    else : 
        x,z =plotOption.P3_OP(float(data["xmin"]),float(data["xmax"]),float(data["ymin"]),float(data["ymax"]),data["x"],data["y"],data["z"])
        return jsonify({"x" :x[0].tolist() , "y" : x[1].tolist(), "z" : z.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

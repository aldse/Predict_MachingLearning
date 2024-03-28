from flask import Flask, Blueprint, render_template, request
from joblib import load
import pandas as pd


app = Flask(__name__)

bp = Blueprint('home', __name__, url_prefix='/home')

@bp.route('/', methods=('GET', 'POST'))
def register():
    return render_template('index.html')

@bp.route('/result', methods=('GET', 'POST'))
def home():
    model = load('../desafio.joblib')

    res = None

    if request.method == 'POST':
        genero = int(request.form['genero'])
        age = int(request.form['age'])
        familiar = int(request.form['familiar'])
        teorcalorico = int(request.form['teorcalorico'])
        vegetais = int(request.form['vegetais'])
        quantasrefeicoes = int(request.form['quantasrefeicoes'])
        entreasrefeicoes = int(request.form['entreasrefeicoes'])
        fumo = int(request.form['fumo'])
        agua = int(request.form['agua'])
        ingestaocalorica = int(request.form['ingestaocalorica'])
        atividadefisica = int(request.form['atividadefisica'])
        telefonetempo = int(request.form['telefonetempo'])
        bebe = int(request.form['bebe'])
        automovel = int(request.form['automovel'])
        bicicleta = int(request.form['bicicleta'])
        moto = int(request.form['moto'])
        urbs = int(request.form['urbs'])
        caminhando = int(request.form['caminhando'])

        resu = model.predict(pd.DataFrame({'Gender': [genero], 
                                           'Age': [age], 
                                           'family_history_with_overweight': [familiar], 
                                           'FAVC': [teorcalorico], 
                                           'FCVC': [vegetais], 
                                           'NCP': [quantasrefeicoes], 
                                           'CAEC': [entreasrefeicoes], 
                                           'SMOKE': [fumo], 
                                           'CH2O': [agua], 
                                           'SCC': [ingestaocalorica], 
                                           'FAF': [atividadefisica], 
                                           'TUE': [telefonetempo], 
                                           'CALC': [bebe], 
                                           'Automobile': [automovel], 
                                           'Bike': [bicicleta], 
                                           'Motorbike': [moto], 
                                           'Public_Transportation': [urbs], 
                                           'Walking': [caminhando]}))


        res = "Obeso" if resu[0] else "Não Obeso"
        
    return render_template('result.html', model=res)


if __name__ == '__main__':
    app.register_blueprint(bp)
    app.run(debug=True)

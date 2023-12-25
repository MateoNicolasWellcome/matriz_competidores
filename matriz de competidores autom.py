# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:42:45 2019

@author: juanp
"""
# Librerías necesarias para llevar a cabo el programa

import math
from math import pi

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import manifold
import openpyxl

# Escalamiento de datos para matriz de competidores
# Introducir la matriz inicial que generan sobre las calificaciones en booking de los atributos
Matriz_atributos = pd.read_excel('Matriz_Atributos.xlsx')
# Matriz_atributos=pd.read_excel(r'G:\Mi unidad\Clientes My Revenue\Analisis\Nuevo MyRevenue Start\Matriz de Competidores\Matriz_Atributos.xlsx')
# Colocar como índices los atributos
Matriz_atributos = Matriz_atributos.set_index('Atributos')
# Guardas los elementos iterativos, en este caso, los hoteles y los atributos
Atributos = Matriz_atributos.index.tolist()
Hoteles = Matriz_atributos.columns.tolist()



# Generar una nueva matriz que vaya  a contener las matrices con las mediciones asociadas al escalamiento de datos
matriz_euclideana = pd.DataFrame(index=range(len(Atributos) * len(Hoteles)), columns=['Atributo', 'Hotel'])
# Adicionar las columnas de atributos y hotel y las combinaciones posibles

k = 0
for i in Atributos:
    for j in Hoteles:
        matriz_euclideana.at[k, 'Atributo'] = i
        matriz_euclideana.at[k, 'Hotel'] = j
        k += 1
# Adicionar la columna asociada a cada uno de los hoteles
for i in Hoteles:
    matriz_euclideana[i] = np.NaN
# Ingresar los datos de distancia euclideana

for i in Hoteles:
    for j in Hoteles:
        for l in Atributos:
            print(i,j,l)
            matriz_euclideana.loc[(matriz_euclideana['Atributo'] == l)
                                  & (matriz_euclideana['Hotel'] == j), i] = abs(
                Matriz_atributos.at[l, i] - Matriz_atributos.at[l, j])
            """matriz_euclideana.at[matriz_euclideana[
                (matriz_euclideana['Atributo'] == l)
                & (matriz_euclideana['Hotel'] == j)]
                .index.values, i] = abs(
                Matriz_atributos.at[l, i] - Matriz_atributos.at[l, j])"""
# generar la matriz de corrdenadas sujeta a que se le introduzcan las coordenadas de los atributos que se pretenden ponderar
print(matriz_euclideana)
Coordenadas_ponderadas = pd.DataFrame()
# Loop para generar las matrices de competidores de cada atributo
for j in Atributos:
    # Escalamiento de datos
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    # Obtener los resultados del escalamiento de datos
    resultados = mds.fit(matriz_euclideana[Hoteles][matriz_euclideana['Atributo'] == j].values)
    # Obtener en una tabla las coordenadas que se ubicarán en el plano cartesiano
    Coordenadas = resultados.embedding_
    # Las coordenadas se volverá en una matriz de dataframe
    Coordenadas = pd.DataFrame(data=Coordenadas[0:], columns=['x', 'y'])
    # Generar un dataframe que va a contener el ponderado de los atributos que se quieren estudiar
    if j in ['Calificación General', 'Precio', 'Distancia', 'Segmento', 'Inventario', 'Servicios']:
        Coordenadas_ponderadas['x_' + j] = Coordenadas['x']
        Coordenadas_ponderadas['y_' + j] = Coordenadas['y']
        # Se asociará cada cordenada a cada uno de los hoteles
    Coordenadas['Hoteles'] = pd.Series(Hoteles)
    # Los hoteles se vuelven indices en la matriz de coordenadas
    Coordenadas = Coordenadas.set_index('Hoteles')
    # Obtener el ratio (que es el radio de una circunferencia) que dertermina que tan cercanos o lejanos se encuentran los competidores
    for i in Coordenadas.index.tolist():
        Coordenadas.at[i, 'Ratio'] = math.sqrt(
            ((Coordenadas.at[Coordenadas.index.tolist()[0], 'x'] - Coordenadas.at[i, 'x']) ** 2) + (
                        (Coordenadas.at[Coordenadas.index.tolist()[0], 'y'] - Coordenadas.at[i, 'y']) ** 2))
    # A partir de este punto se procede a graficar
    # Generar los colores para cada punto
    colors = cm.rainbow(np.linspace(0, 1, len(Coordenadas.index.tolist())))
    # Generar la figura que va a contener el atributo
    Fig = plt.figure(figsize=(14.5, 4))
    # Generar una subgráfica que va a contener el diagrama de dispersión donde se va a graficar cada punto que corresponde a cada hotel
    fig1 = Fig.add_subplot(121)
    # Agregar cada punto a la subgráfica, con el hotel, color y coordenadas que corresponde
    for x, y, l, c, rt in zip(Coordenadas['x'], Coordenadas['y'], Hoteles, colors, Coordenadas['Ratio']):
        fig1.scatter(x, y, label=l, color=c)
        fig1.add_patch(plt.Circle((Coordenadas['x'][0], Coordenadas['y'][0]), rt, color=c, fill=False, linestyle=':',
                                  joinstyle='bevel'))
    # Definir el tamaño de la imagen donde se va a presentar la información
    plt.xlim(min(Coordenadas['x']) - 0.1, max(Coordenadas['x']) + 0.1)
    plt.ylim(min(Coordenadas['y']) - 0.1, max(Coordenadas['y']) + 0.1)
    # Generar y ubicar las convenciones corrrespondientes al lado del gráfico
    handles, labels = fig1.get_legend_handles_labels()
    lgd = fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.35, 1.024))
    # Introducirle un título a la subráfica
    fig1.title.set_text(Hoteles[0] + ',' + ' ' + j)
    # Generar una subgráfica que va a contener el gráfico de radar
    fig2 = Fig.add_subplot(122, polar=True)
    # Colocar la gráfica con le punto 0 en verticañ
    fig2.set_theta_offset(pi / 2)
    fig2.set_theta_direction(-1)
    # Determinar la cantidad de hoteles en la muestra
    categorias = [n / float(len(Hoteles) - 1) * 2 * pi for n in range(len(Hoteles) - 1)]
    categorias += categorias[:1]
    # Marcar los hoteles asociados a cada punto
    plt.xticks(categorias[:-1], Hoteles[1:])
    # Adicionar un utlimo punto de coordenada
    Ratio = Coordenadas['Ratio'].tolist()[1:]
    Ratio += Ratio[:1]
    # Adicionar cada hotel a cada uno de los puntos
    fig2.plot(categorias, Ratio, linestyle='solid')
    fig2.fill(categorias, Ratio, 'b', alpha=0.1)
    # ajustar la gráfica
    plt.tight_layout()
    # Guardar la figura
    # Fig.savefig(r"G:\Mi unidad\Clientes My Revenue\Analisis\Nuevo MyRevenue Start\Matriz de Competidores\-"+Hoteles[0]+"_"+j+"-.png")
    Fig.savefig(f'{Hoteles[0]}_.png')

# En el punto anterior se obtuvieron gráficamente los resultados de los atributos asociados a lo publicado por booking
# A partir de este puto se desarrollará lo necesario para obtener la ubicación espacial de cada uno de lso hoteles en un plano cartesiano a través de geometría euclideana

# En este punto finaliza el cálculo del efecto que tiene la ubicación espacial de los competidores frente
# Introducir la matriz inicial que generan sobre las calificaciones en booking de los atributos
# Matriz_Segmentos=pd.read_excel(r'G:\Mi unidad\Clientes My Revenue\Analisis\Nuevo MyRevenue Start\Matriz de Competidores\Matriz_Segmentos.xlsx')
Matriz_Segmentos = pd.read_excel('Matriz_Segmentos.xlsx')
# Colocar como índices los Segmentos
Matriz_Segmentos = Matriz_Segmentos.set_index('Segmentos')
# lista de segmentos
Segmentos = Matriz_Segmentos.index.tolist()

# Generar una nueva matriz que vaya  a contener las matrices con las mediciones asociadas al escalamiento de datos
Matriz_euc_Segmentos = pd.DataFrame(index=range(len(Segmentos) * len(Hoteles)), columns=['Segmentos', 'Hotel'])
# Adicionar las columnas de atributos y hotel y las combinaciones posibles
k = 0
for i in Segmentos:
    for j in Hoteles:
        Matriz_euc_Segmentos.at[k, 'Segmentos'] = i
        Matriz_euc_Segmentos.at[k, 'Hotel'] = j
        k += 1

# Adicionar la columna asociada a cada uno de los hoteles
for i in Hoteles:
    Matriz_euc_Segmentos[i] = np.NaN
# Ingresar los datos de distancia euclideana
for i in Hoteles:
    for j in Hoteles:
        for l in Segmentos:
            """Matriz_euc_Segmentos.at[Matriz_euc_Segmentos[
                (Matriz_euc_Segmentos['Segmentos'] == l) & (Matriz_euc_Segmentos['Hotel'] == j)].index.values, i] = abs(
                Matriz_Segmentos.at[l, i] - Matriz_Segmentos.at[l, j])"""
            Matriz_euc_Segmentos.loc[(Matriz_euc_Segmentos['Segmentos'] == l)
                                     & (Matriz_euc_Segmentos['Hotel'] == j), i] = abs(
                Matriz_Segmentos.at[l, i] - Matriz_Segmentos.at[l, j])
Coordenadas_Segmentos = pd.DataFrame()
# Loop para generar las gráficas de cada segmento
for j in Segmentos:
    # Escalamiento de datos
    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    # Obtener los resultados del escalamiento de datos
    resultados = mds.fit(Matriz_euc_Segmentos[Hoteles][Matriz_euc_Segmentos['Segmentos'] == j].values)
    # Obtener en una tabla las coordenadas que se ubicarán en el plano cartesiano
    Coordenadas = resultados.embedding_
    # Las coordenadas se volverá en una matriz de dataframe
    Coordenadas = pd.DataFrame(data=Coordenadas[0:], columns=['x', 'y'])
    # Guardar las coordenadas de los segmentos en otra matriz que posteriormente se ponderará
    Coordenadas_Segmentos['x_' + j] = Coordenadas['x']
    Coordenadas_Segmentos['y_' + j] = Coordenadas['y']
    # Se asociará cada cordenada a cada uno de los hoteles
    Coordenadas['Hoteles'] = pd.Series(Hoteles)
    # Los hoteles se vuelven indices en la matriz de coordenadas
    Coordenadas = Coordenadas.set_index('Hoteles')
    # Obtener el ratio (que es el radio de una circunferencia) que dertermina que tan cercanos o lejanos se encuentran los competidores
    for i in Coordenadas.index.tolist():
        Coordenadas.at[i, 'Ratio'] = math.sqrt(
            ((Coordenadas.at[Coordenadas.index.tolist()[0], 'x'] - Coordenadas.at[i, 'x']) ** 2) + (
                        (Coordenadas.at[Coordenadas.index.tolist()[0], 'y'] - Coordenadas.at[i, 'y']) ** 2))
    # A partir de este punto se procede a graficar
    # Generar los colores para cada punto
    colors = cm.rainbow(np.linspace(0, 1, len(Coordenadas.index.tolist())))
    # Generar la figura que va a contener el atributo
    Fig = plt.figure(figsize=(15.5, 4))
    # Generar una subgráfica que va a contener el diagrama de dispersión donde se va a graficar cada punto que corresponde a cada hotel
    fig1 = Fig.add_subplot(121)
    # Agregar cada punto a la subgráfica, con el hotel, color y coordenadas que corresponde
    for x, y, l, c, rt in zip(Coordenadas['x'], Coordenadas['y'], Hoteles, colors, Coordenadas['Ratio']):
        fig1.scatter(x, y, label=l, color=c)
        fig1.add_patch(plt.Circle((Coordenadas['x'][0], Coordenadas['y'][0]), rt, color=c, fill=False, linestyle=':',
                                  joinstyle='bevel'))
    # Definir el tamaño de la imagen donde se va a presentar la información
    plt.xlim(min(Coordenadas['x']) - 0.1, max(Coordenadas['x']) + 0.1)
    plt.ylim(min(Coordenadas['y']) - 0.1, max(Coordenadas['y']) + 0.1)
    # Generar y ubicar las convenciones corrrespondientes al lado del gráfico
    handles, labels = fig1.get_legend_handles_labels()
    lgd = fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.35, 1.024))
    # Introducirle un título a la subráfica
    fig1.title.set_text(Hoteles[0] + ', Segmento de ' + j)
    # Generar una subgráfica que va a contener el gráfico de radar
    fig2 = Fig.add_subplot(122, polar=True)
    # Colocar la gráfica con le punto 0 en verticañ
    fig2.set_theta_offset(pi / 2)
    fig2.set_theta_direction(-1)
    # Determinar la cantidad de hoteles en la muestra
    categorias = [n / float(len(Hoteles) - 1) * 2 * pi for n in range(len(Hoteles) - 1)]
    categorias += categorias[:1]
    # Marcar los hoteles asociados a cada punto
    plt.xticks(categorias[:-1], Hoteles[1:])
    # Adicionar un utlimo punto de coordenada
    Ratio = Coordenadas['Ratio'].tolist()[1:]
    Ratio += Ratio[:1]
    # Adicionar cada hotel a cada uno de los puntos
    fig2.plot(categorias, Ratio, linestyle='solid')
    fig2.fill(categorias, Ratio, 'b', alpha=0.1)
    # ajustar la gráfica
    plt.tight_layout()
    # Fig.savefig(r"G:\Mi unidad\Clientes My Revenue\Analisis\Nuevo MyRevenue Start\Matriz de Competidores\-"+Hoteles[0]+"_Segmento de "+j+"-.png")
    Fig.savefig(f"{Hoteles[0]}_Segmento de {j}-.png")

# Generar el ponderado de los segmentos para determinar los competidores más cercanos o lejanos
for i in range(len(Coordenadas_Segmentos.index.tolist())):
    pond_x = []
    pond_y = []
    for j, k in zip(Matriz_Segmentos[Matriz_Segmentos.columns.tolist()[i]].tolist(), Segmentos):
        pond_x.append(j * Coordenadas_Segmentos.at[i, 'x_' + k])
        pond_y.append(j * Coordenadas_Segmentos.at[i, 'y_' + k])
    Coordenadas_Segmentos.at[i, 'x_Segmento'] = sum(pond_x)
    Coordenadas_Segmentos.at[i, 'y_Segmento'] = sum(pond_y)
# Ingresar los Hoteles
Coordenadas = pd.DataFrame(index=range(len(Hoteles)))
# Se tranfieren los datos de los segmentos ponderados a las coordenadas comunes y corrientes para graficar
Coordenadas['x'] = Coordenadas_Segmentos['x_Segmento']
Coordenadas['y'] = Coordenadas_Segmentos['y_Segmento']
# Obtener el ratio (que es el radio de una circunferencia) que dertermina que tan cercanos o lejanos se encuentran los competidores
for i in Coordenadas.index.tolist():
    Coordenadas.at[i, 'Ratio'] = math.sqrt(
        ((Coordenadas.at[Coordenadas.index.tolist()[0], 'x'] - Coordenadas.at[i, 'x']) ** 2) + (
                    (Coordenadas.at[Coordenadas.index.tolist()[0], 'y'] - Coordenadas.at[i, 'y']) ** 2))
# Generar la figura que va a contener el atributo
Fig = plt.figure(figsize=(15.5, 4))
# Generar una subgráfica que va a contener el diagrama de dispersión donde se va a graficar cada punto que corresponde a cada hotel
fig1 = Fig.add_subplot(121)
# Agregar cada punto a la subgráfica, con el hotel, color y coordenadas que corresponde
for x, y, l, c, rt in zip(Coordenadas['x'], Coordenadas['y'], Hoteles, colors, Coordenadas['Ratio']):
    fig1.scatter(x, y, label=l, color=c)
    fig1.add_patch(plt.Circle((Coordenadas['x'][0], Coordenadas['y'][0]), rt, color=c, fill=False, linestyle=':',
                              joinstyle='bevel'))
# Definir el tamaño de la imagen donde se va a presentar la información
plt.xlim(min(Coordenadas['x']) - 0.1, max(Coordenadas['x']) + 0.1)
plt.ylim(min(Coordenadas['y']) - 0.1, max(Coordenadas['y']) + 0.1)
# Generar y ubicar las convenciones corrrespondientes al lado del gráfico
handles, labels = fig1.get_legend_handles_labels()
lgd = fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.35, 1.024))
# Introducirle un título a la subráfica
fig1.title.set_text(Hoteles[0] + ', Segmentos')
# Generar una subgráfica que va a contener el gráfico de radar
fig2 = Fig.add_subplot(122, polar=True)
# Colocar la gráfica con le punto 0 en verticañ
fig2.set_theta_offset(pi / 2)
fig2.set_theta_direction(-1)
# Determinar la cantidad de hoteles en la muestra
categorias = [n / float(len(Hoteles) - 1) * 2 * pi for n in range(len(Hoteles) - 1)]
categorias += categorias[:1]
# Marcar los hoteles asociados a cada punto
plt.xticks(categorias[:-1], Hoteles[1:])
# Adicionar un utlimo punto de coordenada
Ratio = Coordenadas['Ratio'].tolist()[1:]
Ratio += Ratio[:1]
# Adicionar cada hotel a cada uno de los puntos
fig2.plot(categorias, Ratio, linestyle='solid')
fig2.fill(categorias, Ratio, 'b', alpha=0.1)
# Ajustar la gráfica
plt.tight_layout()
# Guardar la figura
# Fig.savefig(r"G:\Mi unidad\Clientes My Revenue\Analisis\Nuevo MyRevenue Start\Matriz de Competidores\-" + Hoteles[
#    0] + "_Segmentos-.png")
# Agregar la ponderación de segmentos a las coordenadas que se van a ponderar
Fig.savefig(f"{Hoteles[0]}_Segmentos-.png")
for i in ['x', 'y']:
    Coordenadas_ponderadas[i + '_Segmento'] = Coordenadas_Segmentos[i + '_Segmento']
# Definir los criterios a través de los cuales se tendrán en cuenta la ponderación total de competidores
Criterios_de_competencia = ['Calificación General', 'Precio', 'Distancia', 'Segmento', 'Inventario', 'Servicios']
# Generar una matriz con las diferencias absolutas respecto a el hotel de referencia
Dif_Absoluta = pd.DataFrame(index=Coordenadas_ponderadas.index.tolist(),
                            columns=Coordenadas_ponderadas.columns.tolist())
# Agregar la diferencia de cada uno de los campos respecto al hotel de diferencia
for i in Coordenadas_ponderadas.columns.tolist():
    for j in Coordenadas_ponderadas.index.tolist():
        Dif_Absoluta.at[j, i] = abs(Coordenadas_ponderadas.at[0, i] - Coordenadas_ponderadas.at[j, i])
# Generar la matriz de coordenadas estandarizadas
Coordenadas_ponderadas_std = pd.DataFrame(index=Coordenadas_ponderadas.index.tolist(),
                                          columns=Coordenadas_ponderadas.columns.tolist())
# Ingresar los datos estandarizados
for i in Coordenadas_ponderadas.columns.tolist():
    for j in Coordenadas_ponderadas.index.tolist():
        Coordenadas_ponderadas_std.at[j, i] = (Coordenadas_ponderadas.at[j, i] - Coordenadas_ponderadas.at[0, i]) / (
            Dif_Absoluta[i].mean())

# Generar un loop para ingresar los valores que van a ponderar los datos que estamos trabajando
ponderadores = []
for i in Criterios_de_competencia:
    percent = input('Introduzca, como valor decimal, el valor con el cual ponderará el atributo de ' + i + ': ')
    ponderadores.append(float(percent))
# Adicionar el ponderado total de toda la matriz de competidores
for i in Coordenadas_ponderadas_std.index.tolist():
    pond_x = []
    pond_y = []
    for j, k in zip(Criterios_de_competencia, ponderadores):
        pond_x.append(k * Coordenadas_ponderadas_std.at[i, 'x_' + j])
        pond_y.append(k * Coordenadas_ponderadas_std.at[i, 'y_' + j])
    Coordenadas_ponderadas_std.at[i, 'x_Total'] = sum(pond_x)
    Coordenadas_ponderadas_std.at[i, 'y_Total'] = sum(pond_y)

# Se tranfieren los datos de los segmentos ponderados a las coordenadas comunes y corrientes para graficar
Coordenadas['x'] = Coordenadas_ponderadas_std['x_Total']
Coordenadas['y'] = Coordenadas_ponderadas_std['y_Total']
# Obtener el ratio (que es el radio de una circunferencia) que dertermina que tan cercanos o lejanos se encuentran los competidores
for i in Coordenadas.index.tolist():
    Coordenadas.at[i, 'Ratio'] = math.sqrt(
        ((Coordenadas.at[Coordenadas.index.tolist()[0], 'x'] - Coordenadas.at[i, 'x']) ** 2) + (
                    (Coordenadas.at[Coordenadas.index.tolist()[0], 'y'] - Coordenadas.at[i, 'y']) ** 2))
# Generar la figura que va a contener el atributo
Fig = plt.figure(figsize=(15.5, 4))
# Generar una subgráfica que va a contener el diagrama de dispersión donde se va a graficar cada punto que corresponde a cada hotel
fig1 = Fig.add_subplot(121)
# Agregar cada punto a la subgráfica, con el hotel, color y coordenadas que corresponde
for x, y, l, c, rt in zip(Coordenadas['x'], Coordenadas['y'], Hoteles, colors, Coordenadas['Ratio']):
    fig1.scatter(x, y, label=l, color=c)
    fig1.add_patch(plt.Circle((Coordenadas['x'][0], Coordenadas['y'][0]), rt, color=c, fill=False, linestyle=':',
                              joinstyle='bevel'))
# Definir el tamaño de la imagen donde se va a presentar la información
plt.xlim(min(Coordenadas['x']) - 0.1, max(Coordenadas['x']) + 0.1)
plt.ylim(min(Coordenadas['y']) - 0.1, max(Coordenadas['y']) + 0.1)
# Generar y ubicar las convenciones corrrespondientes al lado del gráfico
handles, labels = fig1.get_legend_handles_labels()
lgd = fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.35, 1.024))
# Introducirle un título a la subráfica
fig1.title.set_text(Hoteles[0] + ', Matriz de Competidores')
# Generar una subgráfica que va a contener el gráfico de radar
fig2 = Fig.add_subplot(122, polar=True)
# Colocar la gráfica con le punto 0 en verticañ
fig2.set_theta_offset(pi / 2)
fig2.set_theta_direction(-1)
# Determinar la cantidad de hoteles en la muestra
categorias = [n / float(len(Hoteles) - 1) * 2 * pi for n in range(len(Hoteles) - 1)]
categorias += categorias[:1]
# Marcar los hoteles asociados a cada punto
plt.xticks(categorias[:-1], Hoteles[1:])
# Adicionar un utlimo punto de coordenada
Ratio = Coordenadas['Ratio'].tolist()[1:]
Ratio += Ratio[:1]
# Adicionar cada hotel a cada uno de los puntos
fig2.plot(categorias, Ratio, linestyle='solid')
fig2.fill(categorias, Ratio, 'b', alpha=0.1)
# Ajustar la gráfica
plt.tight_layout()
Fig.savefig(f"{Hoteles[0]}_Matriz de Competidores-.png")
# Fig.savefig(r"G:\Mi unidad\Clientes My Revenue\Analisis\Nuevo MyRevenue Start\Matriz de Competidores\-" + Hoteles[
#  0] + "_Matriz de Competidores-.png")

exit()

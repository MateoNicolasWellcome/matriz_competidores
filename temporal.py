
"""
# Generar matriz de distancia georeferenciada
# Matriz_espacial=pd.read_excel(r'G:\Mi unidad\Clientes My Revenue\Analisis\Nuevo MyRevenue Start\Matriz de Competidores\Matriz_Distancias.xlsx')
Matriz_espacial = pd.read_excel('Matriz_Distancias.xlsx')
# colocar los hoteles como índices
Matriz_espacial = Matriz_espacial.set_index('Hoteles')
# Desarrollar el escalamiento de datos
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=1)
# Hacer el cálculo con los datos
resultados = mds.fit(Matriz_espacial)
# Obtener en una tabla las coordenadas que se ubicarán en el plano cartesiano
Coordenadas = resultados.embedding_
# Las coordenadas se volverá en una matriz de dataframe
Coordenadas = pd.DataFrame(data=Coordenadas[0:], columns=['x', 'y'])
# Agregar las cordenadas a la matriz que posteriormente se ponderará
Coordenadas_ponderadas['x_Distancia'] = Coordenadas['x']
Coordenadas_ponderadas['y_Distancia'] = Coordenadas['y']
# Se asociará cada cordenada a cada uno de los hoteles
Coordenadas['Hoteles'] = pd.Series(Hoteles)
# Los hoteles se vuelven indices en la matriz de coordenadas
Coordenadas = Coordenadas.set_index('Hoteles')
# Obtener el ratio (que es el radio de una circunferencia) que dertermina que tan cercanos o lejanos se encuentran los competidores
for i in Coordenadas.index.tolist():
    Coordenadas.at[i, 'Ratio'] = math.sqrt(
        ((Coordenadas.at[Coordenadas.index.tolist()[0], 'x'] - Coordenadas.at[i, 'x']) ** 2) + (
                    (Coordenadas.at[Coordenadas.index.tolist()[0], 'y'] - Coordenadas.at[i, 'y']) ** 2))
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
fig1.title.set_text(Hoteles[0] + ', Distancia')
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
# Fig.savefig(r"G:\Mi unidad\Clientes My Revenue\Analisis\Nuevo MyRevenue Start\Matriz de Competidores\-"+Hoteles[0]+"_Distancia-.png")
Fig.savefig(f"{Hoteles[0]}_Distancia-.png")
"""
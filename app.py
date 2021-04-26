import streamlit as st #funciones generales de streamlit
from bokeh.plotting import figure as grafica #para mostrar graficas de lineas
import plotnine as p9 #pip install plotnine
from bokeh.models import ColumnDataSource
from PIL import Image #para abrir imagenes
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import pydeck as pdk
import datetime#libreria para usar formatos de fechas 
import json#libreria para usar json
import matplotlib.pyplot as plt

image = Image.open('duck.png')
st.title("Reportes para Mama Duck")
col1, col2, col3 = st.beta_columns([1,45,1])#son columnas que maneja streamlit en la pagina
with col2:
    st.image(image, caption='Mama Duck',width=80)

#importamos el json
with open('ejemplo.json') as file:
    datajson = json.load(file)
jnodes=datajson.get("nodes")
jn1=jnodes[0]#esta variable guarda todo el json





#*
#*
## ESPACIO PARA FUNCIONES GLOBALES, DE ML Y DE IA
## ESPACIO PARA FUNCIONES GLOBALES, DE ML Y DE IA
def obtencionCoords():
  #lat y lon 1 es para coordenadas de llamadas
  #lat  lon2 son para coordenadas de NODOS
  list1=list()
  list2=list()

  for i in range(len(jn1.get('1')[0]['history'])):
    auxlat=jn1.get(str('1'))[0]['history'][i]['localization'][0]['lat']
    auxlon=jn1.get(str('1'))[0]['history'][i]['localization'][0]['long']
    list1.append([auxlat,auxlon])
  for i in range(len(jn1.get('2')[0]['history'])):
    auxlat=jn1.get(str('2'))[0]['history'][i]['localization'][0]['lat']
    auxlon=jn1.get(str('2'))[0]['history'][i]['localization'][0]['long']
    list1.append([auxlat,auxlon])
  for i in range(len(jn1.get('3')[0]['history'])):
    auxlat=jn1.get(str('3'))[0]['history'][i]['localization'][0]['lat']
    auxlon=jn1.get(str('3'))[0]['history'][i]['localization'][0]['long']
    list1.append([auxlat,auxlon])
  for i in range(len(jn1.get('4')[0]['history'])):
    auxlat=jn1.get(str('4'))[0]['history'][i]['localization'][0]['lat']
    auxlon=jn1.get(str('4'))[0]['history'][i]['localization'][0]['long']
    list1.append([auxlat,auxlon])
    list2.append([jn1.get('1')[0]['localization'][0]['lat'],jn1.get('1')[0]['localization'][0]['long']])
    list2.append([jn1.get('2')[0]['localization'][0]['lat'],jn1.get('2')[0]['localization'][0]['long']])
    list2.append([jn1.get('3')[0]['localization'][0]['lat'],jn1.get('3')[0]['localization'][0]['long']])
    list2.append([jn1.get('4')[0]['localization'][0]['lat'],jn1.get('4')[0]['localization'][0]['long']])

  a1=np.array(list1)
  a2=np.array(list2)
  #print(a1.reshape(-2,2))
  df1 = pd.DataFrame(
  a1,
  columns=['lat', 'lon'])
  #return df1, df2
  #print(a1.reshape(-2,2))
  df2 = pd.DataFrame(
  a2,
  columns=['lat', 'lon'])
  #return df1, df2
  return df1,df2


def obtencionlistasJS(numnodo):
  jnemergency=list()
  jnumsoc=list()#cuenta las ocurrencias de horas
  jhours=list()
  jndate=list()
  #para jhours
  #para jnemergency
  for i in range(len(jn1.get(numnodo)[0]['history'])):
    varauxem=jn1.get(numnodo)[0]['history'][i]['emergency']
    jnemergency.append(varauxem)
    varauxda=jn1.get(numnodo)[0]['history'][i]['date']
    jndate.append(varauxda)
    varauxhr=jn1.get(numnodo)[0]['history'][i]['hour']
    varauxhr=int(varauxhr[0:2])
    jhours.append(varauxhr)
  #preproc no repetidos en jhours
  jhoursp=[]
  for item in jhours:
      if item not in jhoursp:
          jhoursp.append(item)
  #para jnumsoc
  for item in jhoursp:
    jnumsoc.append(jhours.count(item))
  #para jndate yconvertir a datetime
  dtdate=list()
  for y in range(len(jndate)):
    dtdate.append(datetime.datetime.strptime(str(jndate[y][0]), "{'year': '%Y',  'month': '%m', 'day': '%d'}"))
  #para listar tipos de emergencia no repetidos y contarlos
  nremerg=[]#emergencias de cada tipo no repetidas
  jnemergency2=list()#una copia para tener una lista de strs
  countemetype=list()#aqui se depositan los numeros de ocurrencia de cada emergencia segun el orden de nremerg
  for a in jnemergency:
    if len(a)>1:
      jnemergency2.append(str(a[0]))
      jnemergency2.append(str(a[1]))
    else:
      jnemergency2.append(str(a[0]))

  for item in jnemergency2:
      if item not in nremerg:
          nremerg.append(item)
  #print(nremerg)
  #print(jnemergency)
  ##print(nremerg)
  #para obtener num de datos
  for item in nremerg:
    countemetype.append(jnemergency2.count(item))
  ##print(countemetype)

  return dtdate,jnemergency2,jhoursp,jnumsoc,nremerg,countemetype
#en este orden devuelve...
#fechas en formato datetime (solo fechas, no horas)
#emergencias por tipo (incluyendo si aplica mas de un tipo de emergencia)formato arreglo de [[],[],...]
#horas en las que se declararon las emergencias (sin repetirse)
#numero de ocurrencias de las horas de jhoursp
#tipo de emergencias en formato string y sin repetirse
#numero de emergencias por cada tipo (correspondiendo al orden presentado en nremerg)
def obtencionlistasJS2():
  typemergency=list()
  jdays=list()#cuenta las ocurrencias de horas
  jhours=list()
  jnums=list()
  #para jhours
  #para jnemergency
  for i in range(len(jn1.get('1')[0]['history'])):
    ad=jn1.get('1')[0]['history'][i]['date']
    jdays.append(ad)
  for i in range(len(jn1.get('2')[0]['history'])):
    ad=jn1.get('2')[0]['history'][i]['date']
    jdays.append(ad)
  for i in range(len(jn1.get('3')[0]['history'])):
    ad=jn1.get('3')[0]['history'][i]['date']
    jdays.append(ad)
  for i in range(len(jn1.get('4')[0]['history'])):
    ad=jn1.get('4')[0]['history'][i]['date']
    jdays.append(ad)
  jdaystab=list()
  for e in jdays:
    if e not in jdaystab:
      jdaystab.append(e)
  for item in jdaystab:
    jnums.append(jdays.count(item))
  dtdate=list()
  for y in range(len(jdaystab)):
    dtdate.append(datetime.datetime.strptime(str(jdaystab[y][0]), "{'year': '%Y',  'month': '%m', 'day': '%d'}"))
  return dtdate,jnums

def regresionLinealNumEmergen(X, Y, numuser,numnodo):
    X = X.reshape(-1, 1)
    # cada elemento solo tiene un feature
    Y = Y.reshape(-1, 1)
    # Modelo de regresion lineal
    # transpuesta de x
    xtran = np.transpose(X)  # primero ahcemos la transpuesta de X

    # Producto punto
    xtran_x = np.linalg.inv(xtran.dot(X))  # inversa del producto punto en la transpuesta de X
    xtran_y = xtran.dot(Y)  # producto punto de la transpuesta de X con la Y
    W = xtran_x.dot(xtran_y)  # producto punto de la transpuesta de X con la transpuesta de Y
    y_pred = X * W  # una variable a predecir utilizando el peso
    plt.scatter(X, Y)
    plt.plot(X, y_pred, color='g')

    test = numuser * W
    plt.scatter(numuser, test, color="y")
    plt.scatter(X, Y)
    plt.plot(X, y_pred, color='g')
    prediccion = W[0] * numuser
    st.write("Segun la hora recibida: ", numuser, ", se predijo este número de emergencias:", round(prediccion[0]),
          "Esos son los numeros de casos probables en el nodo",numnodo)



def plothour_vs_numcalls(x,y,color1):
    p = grafica(
    title='Analisis tipo de emergencias',
    x_axis_label='Horas',
    y_axis_label='Num emergencias')

    p.line(x, y, legend_label='Numero de emergencias', line_width=2,color=color1)
    st.bokeh_chart(p, use_container_width=True)
    #muestra una lámina graficando este tipo de datos

def plottypecalls(x,y,color1):
    p = grafica(
    title='Tipo de emergencias',
    x_axis_label='Emergencia',
    y_axis_label='Cantidad')

    p.line(x, y, legend_label='Numero de emergencias', line_width=2,color=color1)
    st.bokeh_chart(p, use_container_width=True)
    #muestra una lámina graficando este tipo de datos








## FIN ESPACIO PARA FUNCIONES GLOBALES, DE ML Y DE IA
## FIN ESPACIO PARA FUNCIONES GLOBALES, DE ML Y DE IA
#*
#*







nodoseleccionado = st.radio("Seleccione un nodo",
('Mama Duck', 'Nodo 2', 'Nodo 3', 'Nodo 4','General'))
if nodoseleccionado=='Mama Duck':
    st.header('Análisis de los datos del Nodo Mama Duck')
    a,b,c,d,e,f=obtencionlistasJS('1')
#quiero intentar analizar equis tipo de llamada mas probable
    plothour_vs_numcalls(c,d,'red')
#predTipoEmergencia(x,y)
    #if st.button("Estimar no. de emergencias dada una hora", key=None, 
   #help="Este botón lleva a cabo una función de inteligencia artificial, la cual, analizando el número de emergencias y las horas, predice por regresión lineal emergencias estimadas"):
    user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
    for x in user_input:
        regresionLinealNumEmergen(np.array(c), np.array(d), int(x),nodoseleccionado)
        st.success("Se ha completado el análisis para la hora seleccionada")

    

elif nodoseleccionado=='Nodo 2':
    st.header('Análisis de los datos del Nodo 2')
    a,b,c,d,e,f=obtencionlistasJS('2')
#quiero intentar analizar equis tipo de llamada mas probable
    plothour_vs_numcalls(c,d,'orange')
#predTipoEmergencia(x,y)
    #if st.button("Estimar no. de emergencias dada una hora", key=None, 
   #help="Este botón lleva a cabo una función de inteligencia artificial, la cual, analizando el número de emergencias y las horas, predice por regresión lineal emergencias estimadas"):
    user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
    for x in user_input:
        regresionLinealNumEmergen(np.array(c), np.array(d), int(x),nodoseleccionado)
        st.success("Se ha completado el análisis para la hora seleccionada")

elif nodoseleccionado=='Nodo 3':
    st.header('Análisis de los datos del Nodo 3')
    a,b,c,d,e,f=obtencionlistasJS('3')
#quiero intentar analizar equis tipo de llamada mas probable
    plothour_vs_numcalls(c,d,'brown')
#predTipoEmergencia(x,y)
    #if st.button("Estimar no. de emergencias dada una hora", key=None, 
   #help="Este botón lleva a cabo una función de inteligencia artificial, la cual, analizando el número de emergencias y las horas, predice por regresión lineal emergencias estimadas"):
    user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
    for x in user_input:
        regresionLinealNumEmergen(np.array(c), np.array(d), int(x),nodoseleccionado)
        st.success("Se ha completado el análisis para la hora seleccionada")
elif nodoseleccionado=='Nodo 4':
    st.header('Análisis de los datos del Nodo 4')
    a,b,c,d,e,f=obtencionlistasJS('4')
#quiero intentar analizar equis tipo de llamada mas probable
    plothour_vs_numcalls(c,d,'purple')
#predTipoEmergencia(x,y)
    #if st.button("Estimar no. de emergencias dada una hora", key=None, 
   #help="Este botón lleva a cabo una función de inteligencia artificial, la cual, analizando el número de emergencias y las horas, predice por regresión lineal emergencias estimadas"):
    user_input = st.multiselect("Seleccione la hora u horas a predecir",[0.00,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00,21.00,22.00,23.00])
    for x in user_input:
        regresionLinealNumEmergen(np.array(c), np.array(d), int(x),nodoseleccionado)
        st.success("Se ha completado el análisis para la hora seleccionada")
#voy a desplegar el historial dellamadas en una tabla
#voy a mostrar el numero de llamadas por dia

elif nodoseleccionado=='General':
  st.header('Análisis general de la red Mama Duck')


  a,b=obtencionlistasJS2()
  listdt=list()
  for x in range(len(a)):
    stra=str(a[x])
    listdt.append([stra[5:10],b[x]])
  df=pd.DataFrame(listdt,columns=['Fecha','Emergencias'])
  #df["Fecha"]=pd.to_datetime(df["Fecha"])
  #df["Fecha"]=str(df["Fecha"][:])
  #df.sort_values(['Fecha'])
  dotgraph = p9.ggplot(data=df,
                          mapping=p9.aes(x='Fecha', y='Emergencias'))

  st.pyplot(p9.ggplot.draw(dotgraph + p9.geom_point(color='green',alpha=0.5,size=2.7)))
  #voy a mostrar el numero de llamadas por hora dado un dia
  #mostrar estadistica de tipo de accidente--graficas
  #mostrar warnings del nodo--lo que hizo alex de acuerdo con el json recibido
  #mostrar emergencias comunes--haciendo abstraccion de los datos recibidos
  #predecir posibles accidentes totales--usando el algoritmo de IRIS (visto en clases de ML)
  st.header("Mapa de nodos y emergencias")
  st.write("Hexagonos: casos de emergencia, Círculo verde: nodos")
  df1,df2=obtencionCoords()
  st.pydeck_chart(pdk.Deck(
  map_style='mapbox://styles/uberdata/cjoqbbf6l9k302sl96tyvka09',
  initial_view_state=pdk.ViewState(
      latitude=20.63494981128319,
      longitude=-103.40648023281342,
      zoom=17,
      pitch=50,
  ),
  
  layers=[
      pdk.Layer(
          'HexagonLayer',
          data=df2,
          get_position='[lon, lat]',
          #colorRange='[0, 240,255,255]',
          radius=3,
          elevation_scale=4,
          elevation_range=[0, 10],
          pickable=True,
          extruded=True,
      ),
      pdk.Layer(
          'ScatterplotLayer',
          data=df1,
          get_position='[lon, lat]',
          get_color='[100, 230, 0, 160]',
          get_radius=2,
      ),
  ],
  ))



#SIN FUNCIONALIDAD POR AHORA PERO PUEDEN SERVIR MAS ADELANTE
#st.header('Histórico por día')
#st.date_input("Seleccione la fecha para consultar: ", 
#value=None, min_value=datetime.date(2021,4,10), max_value=None, key=None, help="Presione sobre la barra para desplegar el calendario")
    #mostrar emergencias por hora
#st.button("Estimar no. de emergencias dada una hora", key=None,  help="Este botón lleva a cabo una función de inteligencia artificial, la cual, analizando el número de emergencias y las horas, predice por regresión lineal emergencias estimadas")

# arr = np.random.normal(1, 1, size=100)
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)
#
# st.pyplot(fig)

#st.time_input("Escriba la hora a consultar", value=None, key=None, help=None)
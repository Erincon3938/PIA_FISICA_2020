'''

                            Producto Interdisciplinario de Aprendizaje
                            
Autor : Ernesto Guadalupe Rincon Ortiz
Fecha : 09 de juniode 2020

'''

from __future__ import division
from mpl_toolkits.mplot3d import axes3d
from math import pi
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt

def Grafica_RC () :
    
    plt.style.use('ggplot')
    c = 100 * 10**(-6)
    v = float(input("\n Ingrese el Valor del Voltaje (V) : "))
    r = float(input("\n Ingrese el Valor de la Resistencia (Ω) : "))

    t = np.linspace(0,1,1000)

    q = c*v*(1-np.exp((-1/(r*c))*t))
    i = (v/r)*np.exp((-1/(r*c))*t)

    print('\n Tau : ' +str (1/(r*c)))
    print('\b Pico de Corriente (A) : ',v/r)

    plt.plot([0,t[-1]],[c*v,c*v],label='Pico de Carga')
    plt.plot(t,q,label='Carga del Capacitor (C)')
    plt.plot(t,i,label='Corriente (A)')


    plt.xlabel('Tiempo (s)')
    plt.title('Circuito RC')
    plt.legend()
    plt.show()



def Grafica_Campo () :
    
    x = np.linspace(-4,4,10)
    y = np.linspace(-4,4,10)
    z = np.linspace(-4,4,10)

    x,y,z = np.meshgrid(x,y,z)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    def B(x,y):
        i = float(input("\n Ingrese la Intensidad de Corriente (I) : "))
        mu = 1.26 * 10**(-6)
        mag = (mu/(2*np.pi))*(i/np.sqrt((x)**2+(y)**2))
        by = mag * (np.cos(np.arctan2(y,x)))
        bx = mag * (-np.sin(np.arctan2(y,x)))
        bz = z*0
        return bx,by,bz

    def Cilindro(r):

        phi = np.linspace(-2*np.pi,2*np.pi,100)
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        return x,y


    bx,by,bz = B(x,y)
    cx,cy = Cilindro(0.2)


    ax.quiver(x,y,z,bx,by,bz,color='b',length=1,normalize=True)

    for i in np.linspace(-4,4,800):
        ax.plot(cx,cy,i,label='Cilindro',color='r')

    plt.title('Campo Magnetico en una Corriente Rectilinea')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def Biot_Savart () :

    os.system ("cls")
    print ("\t\t\t\t\tLey de Biot-Savart\n")
    print ("\n ¿Que Desea Hacer? \n")
    print ("\n 1) Ley de Biot-Savart ")
    print ("\n 2) Grafica de un Campo Magnetico en una Corriente Rectilinea\n")
    resp3 = int(input("\n Ingrese una Opcion [1-2] : "))
    os.system ("cls")

    while not (resp3 >= 1 and resp3 <= 2) :

        print ("\n Error: Solo Puedes Ingresar un Valor entre 1 o 2\n")
        print ("\n ¿Que Desea Hacer? \n")
        print ("\n 1) Ley de Biot-Savart ")
        print ("\n 2) Grafica de un Campo Magnetico en una Corriente Rectilinea\n")
        resp3 = int(input("\n Ingrese una Opcion [1-2] : "))
        os.system ("cls")

    if resp3 == 1 :

        mu0 = 4e-17 * pi
        print ("\n ¿Cuantos Valores Ingresara? \n")
        print ("\n 1) Un Valor")
        print ("\n 2) Minimos Cuadrados en la Ley de Biot-Svart\n")
        resp2 = int(input("\n Ingrese una Opcion [1-2] : "))
        os.system ("cls")

        while not (resp2 >= 1 and resp2 <= 2) :

            print ("\n Error: Solo Puedes Ingresar un Valor entre 1 o 2\n")
            print ("\n 1) Un Valor")
            print ("\n 2) Minimos Cuadrados en la Ley de Biot-Svart\n")
            resp2 = int(input("\n Ingrese una Opcion [1-2] : "))
            os.system ("cls")

        if resp2 == 1 :

            os.system ("cls")
            i = float(input("\n Ingrese la Intensidad de Corriente (I) : "))
            R = float(input("\n Ingrese la Distancia mas Corta en Linea Recta desde P hasta la Corriente (m) : "))
            B = (mu0 * i) / (2*pi*R)
            print ("\n El Campo Magnetico es : " + str (B) + str (" T") )

        if resp2 == 2 :

            os.system ("cls")
            i = float(input("\n Ingrese la Intensidad de Corriente (I) : "))
            n = 0
            n = int(input("\n Ingrese cuantos valores ingresara: "))

            while not (n >= 2) :

                print ("\n Error : Ingrese un Valor Mayor que 1 \n")
                n = int(input(" Ingrese cuantos valores ingresara: "))

            os.system ("cls")

            r = []
            b = []

            os.system ("cls")

            for i in range (n) :

                R = float(input("\n Ingrese el Valor " + str (i+1) + str (" de la Distancia : ")))
                r.append (R)

            os.system ("cls")

            r = np.array (r)
            b = np.array (b)

            Suma_de_X = sum (r)
            Suma_de_Y = sum ((mu0 * i) / (2*pi*r))
            Suma_de_X2 = sum (r*r)
            Suma_de_Y2 = sum ((mu0 * i) / (2*pi*r) * ((mu0 * i) / (2*pi*r)))
            Suma_de_XY = sum (r*((mu0 * i) / (2*pi*r)))
            Promedio_de_X = Suma_de_X / n
            Promedio_de_Y = Suma_de_Y / n

            m = (Suma_de_X*Suma_de_Y - n*Suma_de_XY) / (Suma_de_X**2 - n*Suma_de_X2)
            b = Promedio_de_Y - m*Promedio_de_X

            Sigma_de_X = np.sqrt (Suma_de_X2/n - Promedio_de_X**2)
            Sigma_de_Y = np.sqrt (Suma_de_Y2/n - Promedio_de_Y**2)
            Sigma_de_XY = Suma_de_XY/n - Promedio_de_X*Promedio_de_Y

            R2 = (Sigma_de_XY/(Sigma_de_X*Sigma_de_Y))

            print ("\n La Pendiente (m) =  " +str (m))
            print ("\n Punto donde corta en el eje y =  " +str (b))

            if m == 0:
                print ("\n La Pendiente es 0 por lo cual no tiene un punto de Intersección")

            else :
                print ("\n Punto donde corta en el eje x = " +str((-(b))/m))

            if b < 0 :
                z = b / (-1)
                print ("\n La Recta Obtenida con el Método de los Mínimos Cuadrados es: ")
                print ("\n\t y = " + str (m) + str ("x - ") + str(z))

            else :
                print ("\n La Recta Obtenida con el Método de los Mínimos Cuadrados es: ")
                print ("\n\t y = " + str (m) + str ("x + ") + str(b))

            print ("\n El Coeficiente de Correlacion es : " + str (R2) + str ("\n"))

            print("\n\t\t\t ¿Que Desea Hacer?\n")
            print ("\n 1) Linealizar los Datos y Mostrar la Grafica de los Datos Linealizados\n")
            print ("\n 2) Ingresar algun Dato en la Ecuacion y Mostrar su Grafica\n ")
            resp = int(input("\n Ingrese una Opcion [1-2]  :  "))

            while not (resp >= 1 and resp <= 2) :
                print("\n Error : Solo Puede Ingresar un Numero del 1 al 2 : \n")
                print ("\n 1) Linealizar los Datos y Mostrar la Grafica de los Datos Linealizados\n")
                print ("\n 2) Ingresar algun Dato en la Ecuacion y Mostrar su Grafica\n ")
                resp = int(input("\n Ingrese una Opcion [1-2]  :  "))
                os.system ("cls")

            if resp == 1 :

                os.system ("cls")
                w = []

                print ("\n ¿Como Desea Linealizar los Datos? \n")
                print ("\n 1) Logaritmo Natural (ln)")
                print ("\n 2) Logaritmo Decimal (log10)\n")
                resp = int(input("\n Ingrese una Opcion [1-2] : "))
                os.system ("cls")

                while not (resp >= 1 and resp <= 2) :

                    print("\n Error : Solo Puede Ingresar un Numero del 1 al 2 : \n")
                    print ("\n\n\t\t\t\tDatos linealizados por Logaritmo Natural\n\n")
                    print ("\n 1) Logaritmo Natural (ln)")
                    print ("\n 2) Logaritmo Decimal (log10)\n")
                    resp = int(input("\n Ingrese una Opcion [1-2] : "))
                    os.system ("cls")

                if resp == 1 :

                    print ("\n\n\t\t\t\tDatos linealizados por Logaritmo Natural (ln)\n\n")
                    w = ((mu0 * i) / (2*pi*r))
                    w = np.log10 (w)
                    r = np.log10 (r)
                    plt.xlabel ('ln (R)')
                    plt.ylabel ('ln (B)')

                else :

                    print ("\n\n\t\t\t\tDatos linealizados por Logaritmo Decimal (log10)\n\n")
                    pi2 = 3.14159265
                    w = ((mu0 * i) / (2*pi*r))
                    w = np.log (w)
                    r = np.log (r)
                    plt.xlabel ('log10 (R)')
                    plt.ylabel ('log10 (B)')


                Suma_de_X = sum (r)
                Suma_de_Y = sum (w)
                Suma_de_X2 = sum (r*r)
                Suma_de_Y2 = sum (w*w)
                Suma_de_XY = sum (r*w)
                Promedio_de_X = Suma_de_X / n
                Promedio_de_Y = Suma_de_Y / n

                m = (Suma_de_X*Suma_de_Y - n*Suma_de_XY) / (Suma_de_X**2 - n*Suma_de_X2)
                b = Promedio_de_Y - m*Promedio_de_X

                Sigma_de_X = np.sqrt (Suma_de_X2/n - Promedio_de_X**2)
                Sigma_de_Y = np.sqrt (Suma_de_Y2/n - Promedio_de_Y**2)
                Sigma_de_XY = Suma_de_XY/n - Promedio_de_X*Promedio_de_Y

                R2 = (Sigma_de_XY/(Sigma_de_X*Sigma_de_Y))

                print ("\n La Pendiente (m) =  " +str (m))
                print ("\n Punto donde corta en el eje y =  " +str (b))

                if m == 0:
                    print ("\n La Pendiente es 0 por lo cual no tiene un punto de Intersección")

                else :
                    print ("\n Punto donde corta en el eje x = " +str((-(b))/m))

                if b < 0 :
                    z = b / (-1)
                    print ("\n La Recta Obtenida con el Método de los Mínimos Cuadrados es: ")
                    print ("\n\t y = " + str (m) + str ("x - ") + str(z))

                else :
                    print ("\n La Recta Obtenida con el Método de los Mínimos Cuadrados es: ")
                    print ("\n\t y = " + str (m) + str ("x + ") + str(b))

                print ("\n El Coeficiente de Correlacion es : " + str (R2) + str ("\n"))

                Enter = input ("\n\n  Presione el Enter Para Poder ver la Grafica \n")
                plt.plot ( r , w , 'o' , color ='orange' , label = 'Datos' )
                plt.plot (r , m*r + b , color = 'red' , label = 'Ajuste')
                plt.title ('Grafica de Regresion Lineal')
                plt.grid ()
                plt.legend ()
                plt.show ()

            elif resp == 2  :

                print ("\n ¿Que dato desea Ingresar? \n ")
                print ("\n\t X ")
                print ("\n\t Y ")
                opc = input("\n Ingrese una Opcion [X-Y] : ")


                while not (opc == 'X' or opc == 'x' or opc == 'Y' or opc == 'Y') :
                    print ("\n Error : Solo Puede Ingresar un Letra ya sea X o Y : \n")
                    print ("\n X ")
                    print ("\n Y ")
                    opc = input("\n Ingrese una Opcion [X-Y]\n")
                    os.system ("cls")

                if (opc == 'X' or opc == 'x') :

                    y2 = float(input("\n Ingrese el Valor de X : "))
                    y2 = m*y2+b
                    print ("\n El Valor de Y es : " + str (y2))

                else :
                    x2 = float(input("\n Ingrese el Valor de Y : "))
                    x2 = (x2-b)/m
                    print ("\n El Valor de Y es : " + str (x2))


                Enter = input ("\n\n  Presione el Enter Para Poder ver la Grafica \n")

                plt.plot ( r , ((mu0 * i) / (2*pi*r)) , 'o' , color ='orange' , label = 'Datos' )
                plt.plot (r , m*r + ((mu0 * i) / (2*pi*r)) , color = 'red' , label = 'Ajuste')
                plt.xlabel ('R')
                plt.ylabel ('B')
                plt.title ('Grafica de Regresion Lineal')
                plt.grid ()
                plt.legend ()
                plt.show ()


    if resp3 == 2 :

        Grafica_Campo ()


def Error_Relativo () :

    os.system ("cls")
    print ("\t\t\t\t\t\t\tError Relativo\n")
    n = 0
    n = int(input(" Ingrese cuantos valores ingresara: "))

    while not (n >= 2) :

        print ("\n Error : Ingrese un Valor Mayor que 1 \n")
        n = int(input(" Ingrese cuantos valores ingresara: "))

    os.system ("cls")

    q = []

    for i in range (n) :

        Q = float(input("\n Ingrese el Valor " + str (i+1) + str (" : ")))
        q.append (Q)

    q = np.array (q)
    os.system ("cls")

    Suma_de_q = sum (q)
    Promedio = Suma_de_q / n
    Error_Estimado = (sum(abs(q-Promedio))) / n
    Error_Relativo = Error_Estimado / Promedio *100

    print ("\n Promedio : " +str (Promedio))
    print ("\n Error Estimado : " +str (Error_Estimado))
    print ("\n Error Relativo : " + str (Error_Relativo))


def Calculadora_Capacitores () :

    os.system ("cls")
    print ("\t\t\t\t\tCalculadora de Capaitores Paralela y Serial\n")
    print ("\n 1) Capaitores en Serie ")
    print ("\n 2) Capaitores en Paralelo\n")
    resp = int(input("\n Ingrese una Opcion [1-2] : "))
    os.system ("cls")

    while not (resp >= 1 and resp <= 2) :

        print("\n Error : Solo Puede Ingresar un Numero del 1 al 2 : \n")
        print ("\n ¿Que Desea Calcular? \n")
        print ("\n 1) Capaitores en Serie ")
        print ("\n 2) Capaitores en Paralelo\n")
        resp = int(input("\n Ingrese una Opcion [1-2] : "))
        os.system ("cls")

    n = 0
    n = int(input(" Ingrese cuantos valores ingresara: "))

    while not (n >= 2) :

        print ("\nError : Ingrese un Valor Mayor que 1 \n")
        n = int(input(" Ingrese cuantos valores ingresara: "))
        os.system ("cls")

    c = []

    os.system ("cls")

    for i in range (n) :

        C = float(input("\n Ingrese el Valor " + str (" de la Resistencia ") + str (i+1) + str (" : ")))
        c.append (C)

    c = np.array (c)
    os.system ("cls")

    if resp ==  1 :

        Suma_de_C = sum (1/c)
        Suma_de_C = 1 / Suma_de_C
        os.system ("cls")
        print ("\n Capacitancia Total en Serie : " + str (Suma_de_C) + str (" F"))

    if resp ==  2 :

        Suma_de_C = sum (c)
        os.system ("cls")
        print ("\n Capacitancia Total Paralela : " + str (Suma_de_C) + str (" F"))




def Calculadora_Resistencias () :

    os.system ("cls")
    print ("\t\t\t\t\tCalculadora de Resistencia Paralela y Serial\n")
    print ("\n ¿Que Desea Calcular? \n")
    print ("\n 1) Resistencias en Paralelo ")
    print ("\n 2) Resistencias en Serie\n")
    resp = int(input("\n Ingrese una Opcion [1-2] : "))
    os.system ("cls")

    while not (resp >= 1 and resp <= 2) :

        print("\n Error : Solo Puede Ingresar un Numero del 1 al 2 : \n")
        print ("\n ¿Que Desea Calcular? \n")
        print ("\n 1) Resistencias en Paralelo ")
        print ("\n 2) Resistencias en Serie\n")
        resp = int(input("\n Ingrese una Opcion [1-2] : "))
        os.system ("cls")

    n = 0
    n = int(input(" Ingrese cuantos valores ingresara: "))

    while not (n >= 2) :

        print ("\nError : Ingrese un Valor Mayor que 1 \n")
        n = int(input(" Ingrese cuantos valores ingresara: "))
        os.system ("cls")

    r = []

    os.system ("cls")

    for i in range (n) :

        R = float(input("\n Ingrese el Valor " + str (" de la Resistencia ") + str (i+1) + str (" : ")))
        r.append (R)

    r = np.array (r)
    os.system ("cls")

    if resp ==  1 :

        Suma_de_R = sum (1/r)
        Suma_de_R = 1 / Suma_de_R
        os.system ("cls")
        print ("\n Resistencia Total Paralela : " + str (Suma_de_R) + str (" Ω"))

    if resp ==  2 :

        Suma_de_R = sum (r)
        os.system ("cls")
        print ("\n Resistencia Total en Serie : " + str (Suma_de_R) + str (" Ω"))




def Lineas_Campo () :
    
    os.system ("cls")

    def Phi( r, rp, q ):
        phi = 1/(4*np.pi*eps0)*q/np.linalg.norm( r - rp )
        return phi

    def Campo_Electrico( r, rp, q ):
        E = 1/(4*np.pi*eps0)*q*(r - rp)/np.linalg.norm( r - rp )**3
        return E

    eps0 = 8.85418781762e-12
    Nres = 25
    Xarray = np.linspace( 0, 10, Nres )
    Yarray = np.linspace( 0, 10, Nres )
    X, Y = plt.meshgrid( Xarray, Yarray )

    q1 = float (input("Ingrese el el Valor de la Carga 1 : "))
    rp1 = np.array( [4,4] )
    phi1 = np.zeros( (Nres,Nres) )
    E1x = np.ones( (Nres,Nres) )
    E1y = np.ones( (Nres,Nres) )


    for i in range(Nres):

        for j in range(Nres):

            r = np.array( [Xarray[i], Yarray[j]] )
            phi1[i,j] = Phi( r, rp1, q1 )
            E = Campo_Electrico( r, rp1, q1 )
            E1x[i,j], E1y[i,j] = E/np.linalg.norm(E)


    q2 = float (input("\nIngrese el el Valor de la Carga 2 : "))
    os.system ("cls")
    rp2 = np.array( [6,6] )
    phi2 = np.zeros( (Nres,Nres) )
    E2x = np.ones( (Nres,Nres) )
    E2y = np.ones( (Nres,Nres) )
    
    for i in range(Nres):

        for j in range(Nres):
            r = np.array( [Xarray[i], Yarray[j]] )
            phi2[i,j] = Phi( r, rp2, q2 )
            E = Campo_Electrico( r, rp2, q2 )
            E2x[i,j], E2y[i,j] = E/np.linalg.norm(E)

    os.system ("cls")

    phi_tot = phi1 + phi2
    Ex_tot = np.ones( (Nres,Nres) )
    Ey_tot = np.ones( (Nres,Nres) )
    for i in range(Nres):

        for j in range(Nres):

            E = np.array( [E1x[i,j] + E2x[i,j], \
            E1y[i,j] + E2y[i,j]] )
            E = E/np.linalg.norm(E)
            Ex_tot[i,j], Ey_tot[i,j] = E

    os.system ("cls")

    plt.contour(X, Y, phi_tot, 100)
    plt.quiver( X, Y, Ey_tot, Ex_tot)

    plt.xlim( (0,10) )
    plt.ylim( (0,10) )
    plt.legend()
    plt.title ('Lineas de campo y Equipotenciales de Cargas Puntuales\n')
    os.system ("cls")
    plt.show()

def Codigo_de_Colores () :

    os.system("cls")

    Colores_Resistencias = {

    "negro" : "0" ,
    "marron" : "1" ,
    "rojo" : "2" ,
    "naranja" : "3" ,
    "amarillo" : "4" ,
    "verde" : "5" ,
    "azul" : "6" ,
    "violeta" : "7" ,
    "gris" : "8" ,
    "blanco" : "9"

    }

    print ("\n\t\t\t\t\t\t\tColores : ")
    print ("\n\t\tnegro , marron , rojo , naranja , amarrillo , verde , azul , violeta , gris , blanco\n")

    color_banda01 = input ("\n Ingrese el Color de la Banda 1 : ")
    color_banda02 = input ("\n Ingrese el Color de la Banda 2 : ")
    color_banda03 = input ("\n Ingrese el Color de la Banda 3 : ")

    color_banda01 = color_banda01.lower ()
    color_banda02 = color_banda02.lower ()
    color_banda03 = color_banda03.lower ()

    if (color_banda01 in Colores_Resistencias and color_banda02 in Colores_Resistencias and color_banda03 in Colores_Resistencias) :

        banda = (Colores_Resistencias [color_banda01] , Colores_Resistencias [color_banda02] ,Colores_Resistencias [color_banda03])

        if (int (banda [0]) > 0 and int (banda [2]) < 10) :

            valor = int (banda [0] +banda [1] +"0" *int (banda [2]))
            print ("\n\n El Valor de la Resistencia es de " + str (valor) + str (" Ohms"))

        else :
            print ("\n Error al Ingresar los Datos")

    else :
        print ("\n Error en el Ingreso de la Banada de Colores")




def Campo_Electrico () :

    os.system ("cls")
    print ("\t\t\t\t\t\tCampo Electrico\n")
    print ("\n ¿Que desea encontrar?\n")
    print ("\n 1) Campo Electrico (E)\n")
    print ("\n 2) Fuerza Electrica (F)\n")
    print ("\n 3) Carga (q)\n")
    print ("\n 4) Distancia de la Carga al Punto donde se Mide el Campo\n")
    op = int(input("\n Ingrese una Opcion : "))
    os.system ("cls")

    while not (op >=1 or op <=4) :
        print ("\n Error : Solo Puedes Ingresar una Opcion del 1 al 4 : ")
        print ("\n ¿Que desea encontrar?\n 1) Campo Electrico (E)\n 2) Fuerza Electrica (F)\n 3) Carga (q)\n 4) Distancia de la Carga al Punto donde se Mide el Campo\n")
        op = int(input("\n Ingrese una Opcion : "))
        os.system ("cls")

    if op == 1 :

        print (" ¿Que Forma Desea Utilizar?\n")
        print ("\n 1) E = F/q\n")
        print ("\n 2) E = kq/r²\n")
        op = int(input("\n Ingrese una Opcion : "))
        os.system ("cls")

        while not (op>= 1 or op<=2) :

            print ("\n Error : Solo Puedes Ingresar una Opcion del 1 al 2 : \n")
            print (" ¿Que Forma Desea Utilizar?\n")
            print ("\n 1) E = F/q\n")
            print ("\n 2) E = kq/r²\n")
            op = int(input("\n Ingrese una Opcion : "))
            os.system ("cls")

        if op == 1 :

            F = float(input("\n Ingrese la Fuerza Electrica (F) : "))
            q = float(input("\n Ingrese la Carga Electrica (q) : "))

            while not (F > 0 and q > 0) :

                print ("\n Error : Solo Puedes Ingresar Valores Mayores a Cero \n")
                F = float(input("\n Ingrese la Fuerza Electrica (F) : "))
                q = float(input("\n Ingrese la Carga Electrica (q) : "))


            E = F / q

            if E != int (E) :
                print ("\n El Resultado es : " + str("{: .4f}".format (E)) + str (" N/C"))

            else :
                print ("\n El Resultado es : " + str(int(E)) + str (" N/C"))

        if op == 2 :

            k = 8.854181762e-12
            r = float(input("\n Ingrese la Distancia (r) : "))
            q = float(input("\n Ingrese la Carga Electrica (q) : "))

            while not (q > 0 and r > 0) :

                print ("\n Error : Solo Puedes Ingresar Valores Mayores a Cero \n")
                q = float(input("\n Ingrese la Distancia (r) : "))
                r = float(input("\n Ingrese la Carga Electrica (q) : "))

            E2 = k*q / r**2

            print ("\n El Resultado es : " + str(E2) + str (" N/C"))
            op = 0

    if op == 2 :

        E = float(input("\n Ingrese el valor del Campo Electrico (E) : "))
        q = float(input("\n Ingrese la Carga Electrica (q) : "))

        while not (E > 0 and q > 0) :

            print ("\n Error : Solo Puedes Ingresar Valores Mayores a Cero \n")
            E = float(input("\n Ingrese el valor del Campo Electrico (E) : "))
            q = float(input("\n Ingrese la Carga Electrica (q) : "))

        F = E*q

        if F != int (F) :
            print ("\n El Resultado es : " + str("{: .4f}".format (F)) + str (" N"))

        else :
            print ("\n El Resultado es : " + str(int(F)) + str (" N"))

    if op == 3 :

        F = float(input("\n Ingrese la Fuerza Electrica (F) : "))
        E = float(input("\n Ingrese la Campo Electrica (E) : "))

        while not (F > 0 and E > 0) :

            print ("\n Error : Solo Puedes Ingresar Valores Mayores a Cero \n")
            F = float(input("\n Ingrese la Fuerza Electrica (F) : "))
            E = float(input("\n Ingrese la Campo Electrica (E) : "))

        q = F/E

        if q != int (q) :
            print ("\n El Resultado es : " + str("{: .4f}".format (q)) + str (" C"))

        else :
            print ("\n El Resultado es : " + str(int(q)) + str (" C"))

    if op == 4 :

        k = 8.854181762e-12
        E = float(input("\n Ingrese el valor del Campo Electrico (E) : "))
        q = float(input("\n Ingrese la Carga Electrica (q) : "))

        while not (E > 0 and q > 0) :

            print ("\n Error : Solo Puedes Ingresar Valores Mayores a Cero \n")
            E = float(input("\n Ingrese el valor del Campo Electrico (E) : "))
            q = float(input("\n Ingrese la Carga Electrica (q) : "))

        r =  (k*q/E)**(1/2)
        print ("\n El Resultado es : " + str(r) + str (" m"))


def Ley_de_Ohm () :

    os.system ("cls")

    resp = "Si"

    while (resp == "Si" or resp == "si" or resp == "s" or resp == "S") :
        print ("\t\t\t\t\t\t\tLey de Ohm\n")
        print ("\t\"La intensidad de corriente que circula por un conductor es directamente proporcional a la diferencia\n\t de potencial que existe entre sus extremos e inversamente proporcional a su resistencia eléctrica.\"")
        print ("\n\n ¿Que Desea Obtener?\n")
        print ("\n 1) Corriente (I)\n")
        print ("\n 2) Voltaje (V)\n")
        print ("\n 3) Resistencia (R)\n")
        op = int(input("\n Ingrese una Opcion : "))
        os.system ("cls")

        while not (op >= 1 and op <=3) :
            print ("\n Error : Solo Puede Ingresar un Numero del 1 al 3 : ")
            print ("\n 1) Corriente (I)\n 2) Voltaje (V)\n 3) Resistencia (R)\n")
            op = int (input("\n Ingrese una Opcion : "))
            os.system ("cls")

        if op == 1 :
            V = float(input("\n Ingrese el Voltaje (V) : "))
            R = float(input("\n Ingrese el Valor de la Resistencia (R) : "))

            if R <= 0 :
                print ("\n Error la Resistencia no Puede Tomar el Valor de Cero o ser Menor a Cero , Ingrese otro Valor : ")
                R = float(input("\n Ingrese el Valor de la Resistencia (R) : "))

            else :
                I = V/R


            if I != int (I) :
                print ("\n El Resultado es : " + str("{: .4f}".format (I)) + str (" Amperes"))

            else :
                print ("\n El Resultado es : " + str(int(I)) + str (" Amperes"))


        if op == 2 :
            I = float(input("\n Ingrese el Valor de la Corriente (I) : "))
            R = float(input("\n Ingrese el Valor de la Resistencia (R) : "))

            if R <= 0 :
                print ("\n Error la Resistencia no Puede Tomar el Valor de Cero o ser Menor a Cero , Ingrese otro Valor : ")
                R = float(input("\n Ingrese el Valor de la Resistencia (R) : "))

            else :
                V = I*R


            if V != int (V) :
                print ("\n El Resultado es : " + str("{: .4f}".format (V)) + str (" Voltios"))

            else :
                print ("\n El Resultado es : " + str(int(V)) + str (" Voltios"))

        if op == 3 :
            V = float(input("\n Ingrese el Voltaje (V) : "))
            I = float(input("\n Ingrese el Valor de la Corriente (I) : "))

            if I <= 0 :
                print ("\n Error la Corriente no Puede Tomar el Valor de Cero o ser Menor a Cero , Ingrese otro Valor : ")
                R = float(input("\n Ingrese el Valor de la Corriente (I) : "))

            else :
                R = V/I


            if V != int (R) :
                print ("\n El Resultado es : " + str("{: .4f}".format (R)) + str (" Ohms"))

            else :
                print ("\n El Resultado es : " + str(int(R)) + str (" Ohms"))


        resp = input("\n\n ¿Desea Obtener Obtener otro Dato [Si-No]? : ")
        os.system ("cls")



def Ley_de_Coulomb () :

    os.system ("cls")
    k = 8.854181762e-12
    print ("\t\t\t\t\t\t\tLey de Coulomb\n")
    print ("\t\"La fuerza eléctrica con la que se atraen o repelen dos cargas puntuales en reposo es directamente\n\t proporcional al producto de las mismas,inversamente proporcional al cuadrado de la distancia que\n\t\t\t las separa y actúa en la dirección de la recta que las une.\"")
    print ("\n\n\t¿Que Desea Obtener?\n")
    print ("\n 1) Fuerza Electrica (F) ")
    print ("\n 2) Carga Electrica (q)")
    print ("\n 3) Distancia que Separa las Cargas (r)")
    op = int(input("\n Ingrese una Opcion : "))
    os.system ("cls")

    while not (op >= 1 and op <=3) :
        print ("\n Error : Solo Puede Ingresar un Numero del 1 al 3 : ")
        print ("\n 1) 1) Fuerza Electrica (F)\n 2) Carga Electrica (q)\n Distancia que Separa las Cargas (r)\n")
        op = int (input("\n Ingrese una Opcion : "))
        os.system ("cls")

    if op == 1 :

        q = float(input("\n Ingrese el Valor de Carga Electrica (q1) : "))
        q = abs (q)
        Q = float(input("\n Ingrese el Valor de Carga Electrica (q2) : "))
        Q = abs (Q)
        r = float(input("\n Ingrese el Valor de la Distancia que Separa las Cargas (r) : "))

        while not (r > 0) :

            print ("\n Error : Solo Puedes Ingresar Valores Mayores a Cero \n")
            F = float(input("\n Ingrese el Valor de la Fuerza Electrico (F) : "))
            r = print ("\n Ingrese el Valor de la Distancia que Separa las Cargas (r) : ")
            os.system ("cls")

        F = (k*q*Q) / (r**2)

        print ("\n La Fuerza Electrica es : " + str (F) + " N")

    if op == 2 :

        q = float(input("\n Ingrese el Valor de Carga Electrica (q1) : "))
        q = abs (q)
        F = float(input("\n Ingrese el Valor de la Fuerza Electrica (F)  : "))
        r = float(input("\n Ingrese el Valor de la Distancia que Separa las Cargas (r) : "))

        while not (r > 0 and F > 0) :

            print ("\n Error : Solo Puedes Ingresar Valores Mayores a Cero \n")
            F = float(input("\n Ingrese el Valor de la Fuerza Electrico (F) : "))
            r = print ("\n Ingrese el Valor de la Distancia que Separa las Cargas (r) : ")
            os.system ("cls")

        Q = (F*r**2) / (k*q)

        print ("\n La Carga Electrica es : " + str (Q) + " C")

    if op == 3 :

        q = float(input("\n Ingrese el Valor de Carga Electrica (q1) : "))
        q = abs (q)
        Q = float(input("\n Ingrese el Valor de Carga Electrica (q2) : "))
        Q = abs (Q)
        F = float(input("\n Ingrese el Valor de la Fuerza Electrica (F)  : "))


        while not ( F > 0) :

            print ("\n Error : Solo Puedes Ingresar Valores Mayores a Cero \n")
            F = float(input("\n Ingrese el Valor de la Fuerza Electrico (F) : "))

            os.system ("cls")

        r =  ( k*q*Q / F )^(1/2)

        print ("\n La Distancia es : " + str (r) + " C")

def Linealizacion ( x , y , n ) :

    z = []
    w = []

    print ("\n ¿Como Desea Linealizar los Datos? \n")
    print ("\n 1) Logaritmo Natural (ln)")
    print ("\n 2) Logaritmo Decimal (log10)\n")
    resp = int(input("\n Ingrese una Opcion [1-2] : "))
    os.system ("cls")

    while not (resp >= 1 and resp <= 2) :

        print("\n Error : Solo Puede Ingresar un Numero del 1 al 2 : \n")
        print ("\n\n\t\t\t\tDatos linealizados por Logaritmo Natural\n\n")
        print ("\n 1) Logaritmo Natural (ln)")
        print ("\n 2) Logaritmo Decimal (log10)\n")
        resp = int(input("\n Ingrese una Opcion [1-2] : "))
        os.system ("cls")

    if resp == 1 :

        print ("\n\n\t\t\t\tDatos linealizados por Logaritmo Natural (ln)\n\n")
        z = np.log (x)
        w = np.log (y)
        plt.xlabel ('ln (x)')
        plt.ylabel ('ln (y)')

    else :

        print ("\n\n\t\t\t\tDatos linealizados por Logaritmo Decimal (log10)\n\n")
        z = np.log10 (x)
        w = np.log10 (y)
        plt.xlabel ('log10 (x)')
        plt.ylabel ('log10 (y)')


    Suma_de_X = sum (z)
    Suma_de_Y = sum (w)
    Suma_de_X2 = sum (z*z)
    Suma_de_Y2 = sum (w*w)
    Suma_de_XY = sum (z*w)
    Promedio_de_X = Suma_de_X / n
    Promedio_de_Y = Suma_de_Y / n

    m = (Suma_de_X*Suma_de_Y - n*Suma_de_XY) / (Suma_de_X**2 - n*Suma_de_X2)
    b = Promedio_de_Y - m*Promedio_de_X

    Sigma_de_X = np.sqrt (Suma_de_X2/n - Promedio_de_X**2)
    Sigma_de_Y = np.sqrt (Suma_de_Y2/n - Promedio_de_Y**2)
    Sigma_de_XY = Suma_de_XY/n - Promedio_de_X*Promedio_de_Y

    R2 = (Sigma_de_XY/(Sigma_de_X*Sigma_de_Y))

    print ("\n La Pendiente (m) =  " +str (m))
    print ("\n Punto donde corta en el eje y =  " +str (b))

    if m == 0:
        print ("\n La Pendiente es 0 por lo cual no tiene un punto de Intersección")

    else :
        print ("\n Punto donde corta en el eje x = " +str((-(b))/m))

    if b < 0 :
        z = b / (-1)
        print ("\n La Recta Obtenida con el Método de los Mínimos Cuadrados es: ")
        print ("\n\t y = " + str (m) + str ("x - ") + str(z))

    else :
        print ("\n La Recta Obtenida con el Método de los Mínimos Cuadrados es: ")
        print ("\n\t y = " + str (m) + str ("x + ") + str(b))

    print ("\n El Coeficiente de Correlacion es : " + str (R2) + str ("\n"))

    Enter = input ("\n\n  Presione el Enter Para Poder ver la Grafica \n")
    plt.plot ( z , w , 'o' , color ='orange' , label = 'Datos' )
    plt.plot (z , m*z + b , color = 'red' , label = 'Ajuste')
    plt.title ('Grafica de Regresion Lineal')
    plt.grid ()
    plt.legend ()
    plt.show ()

def Minimos_Cuadrados () :

    os.system ("cls")
    print ("\t\t\t\t\t\tMétodo de Minímos Cuadrados\n")
    n = 0
    n = int(input(" Ingrese cuantos valores ingresara: "))

    while not (n >= 2) :

        print ("\n Error : Ingrese un Valor Mayor que 1 \n")
        n = int(input(" Ingrese cuantos valores ingresara: "))

    os.system ("cls")

    x = []
    y = []


    for i in range (n) :

        X = float(input(" Ingrese el Valor " + str (i+1) + str (" de X : ")))
        x.append (X)

        Y = float(input(" Ingrese el Valor " + str (i+1) + str (" de Y : ")))
        y.append (Y)
        print ("\n")

    os.system ("cls")

    x = np.array (x)
    y = np.array (y)

    Suma_de_X = sum (x)
    Suma_de_Y = sum (y)
    Suma_de_X2 = sum (x*x)
    Suma_de_Y2 = sum (y*y)
    Suma_de_XY = sum (x*y)
    Promedio_de_X = Suma_de_X / n
    Promedio_de_Y = Suma_de_Y / n

    m = (Suma_de_X*Suma_de_Y - n*Suma_de_XY) / (Suma_de_X**2 - n*Suma_de_X2)
    b = Promedio_de_Y - m*Promedio_de_X

    Sigma_de_X = np.sqrt (Suma_de_X2/n - Promedio_de_X**2)
    Sigma_de_Y = np.sqrt (Suma_de_Y2/n - Promedio_de_Y**2)
    Sigma_de_XY = Suma_de_XY/n - Promedio_de_X*Promedio_de_Y

    R2 = (Sigma_de_XY/(Sigma_de_X*Sigma_de_Y))

    print ("\n La Pendiente (m) =  " +str (m))
    print ("\n Punto donde corta en el eje y =  " +str (b))

    if m == 0:
        print ("\n La Pendiente es 0 por lo cual no tiene un punto de Intersección")

    else :
        print ("\n Punto donde corta en el eje x = " +str((-(b))/m))

    if b < 0 :
        z = b / (-1)
        print ("\n La Recta Obtenida con el Método de los Mínimos Cuadrados es: ")
        print ("\n\t y = " + str (m) + str ("x - ") + str(z))

    else :
        print ("\n La Recta Obtenida con el Método de los Mínimos Cuadrados es: ")
        print ("\n\t y = " + str (m) + str ("x + ") + str(b))

    print ("\n El Coeficiente de Correlacion es : " + str (R2) + str ("\n"))

    print("\n\t\t\t ¿Que Desea Hacer?\n")
    print ("\n 1) Linealizar los Datos y Mostrar la Grafica de los Datos Linealizados\n")
    print ("\n 2) Ingresar algun Dato en la Ecuacion y Mostrar su Grafica\n ")
    resp = int(input("\n Ingrese una Opcion [1-2]  :  "))

    while not (resp >= 1 and resp <= 2) :
        print("\n Error : Solo Puede Ingresar un Numero del 1 al 2 : \n")
        print ("\n 1) Linealizar los Datos y Mostrar la Grafica de los Datos Linealizados\n")
        print ("\n 2) Ingresar algun Dato en la Ecuacion y Mostrar su Grafica\n ")
        resp = int(input("\n Ingrese una Opcion [1-2]  :  "))
        os.system ("cls")

    if resp == 1 :

        os.system ("cls")
        Linealizacion (x , y , n)

    elif resp == 2  :

        print ("\n ¿Que dato desea Ingresar? \n ")
        print ("\n\t X ")
        print ("\n\t Y ")
        opc = input("\n Ingrese una Opcion [X-Y] : ")


        while not (opc == 'X' or opc == 'x' or opc == 'Y' or opc == 'Y') :
            print ("\n Error : Solo Puede Ingresar un Letra ya sea X o Y : \n")
            print ("\n X ")
            print ("\n Y ")
            opc = input("\n Ingrese una Opcion [X-Y]\n")
            os.system ("cls")

        if (opc == 'X' or opc == 'x') :

            y2 = float(input("\n Ingrese el Valor de X : "))
            y2 = m*y2+b
            print ("\n El Valor de Y es : " + str (y2))

        elif (opc == 'Y' or opc == 'y') :

            x2 = float(input("\n Ingrese el Valor de Y : "))
            x2 = (x2-b)/m
            print ("\n El Valor de X es : " + str (x2))


        Enter = input ("\n\n  Presione el Enter Para Poder ver la Grafica \n")

        plt.plot ( x , y , 'o' , color ='orange' , label = 'Datos' )
        plt.plot (x , m*x + b , color = 'red' , label = 'Ajuste')
        plt.xlabel ('x')
        plt.ylabel ('y')
        plt.title ('Grafica de Regresion Lineal')
        plt.grid ()
        plt.legend ()
        plt.show ()




def Menu_Principal () :
    print ("\t\t\t\t\tProducto Interdisciplinario de Aprendizaje\n")
    respuesta = input(" ¿Desea resolver un problema [Si/No]? : ")
    os.system ("cls")

    while  not (respuesta == "Si" or respuesta == "si" or respuesta == "No" or respuesta == "no") :
        print (" Eror : Solo puedes ingresar Si o No")
        respuesta = input (" ¿Desea resolver un problema [Si/No]? : ")
        os.system ("cls") ;

    while respuesta == "Si" or respuesta == "si" :
        print ("\t\t\t\t\t\t\tMenu")
        print(" ¿Que desea hacer?\n")
        print ("\n 1) Campo Electrico\n 2) Metodo de Minimos Cuadrados \n 3) Ley de Coulomb\n 4) Ley de Ohm \n 5) Codigo de Colores en Resistencias")
        print (" 6) Calculadora de Resistencia Paralela y Serial\n 7) Calculadora de Capacitor Paralelo y Serial \n 8) Obtener el Error Relativo")
        print (" 9) Ley de Biot-Savart\n 10) Grafica Circuito RC")
        numero = int(input("\n Ingrese una Opcion [1-10] : "))

        while not (numero >= 1 and numero <= 10) :
            print ("\n Error : Ingrese un Numero del 1 al 10")
            print ("\n 1) Campo Electrico\n 2) Metodo de Minimos Cuadrados \n 3) Ley de Coulomb\n 4) Ley de Ohm \n 5) Codigo de Colores en Resistencias")
            print (" 6) Calculadora de Resistencia Paralela y Serial\n 7) Calculadora de Capacitor Paralelo y Serial\n 8) Obtener el Error Relativo")
            print (" 9) Ley de Biot-Savart\n 10) Grafica Circuito RC")
            numero = int(input("\n Ingrese una Opcion [1-10] : "))

        os.system ("cls")

        if numero == 1 :

            resp = "Si"
            os.system ("cls")
            while (resp == "Si" or resp == "si" or resp == "s" or resp == "S") :

                print ("\n ¿Que Desea Realizar?\n")
                print ("\n 1) Calcular el Campo Electrico\n")
                print ("\n 2) Grafica de las Lineas de campo y Equipotenciales de Cargas Puntuales\n")
                op = int(input("\n Ingrese una Opcion : "))
                os.system ("cls")

                while not (op>= 1 or op<=2) :
                    print ("\n Error : Solo Puedes Ingresar una Opcion del 1 al 2 : \n")
                    print ("\n 1) Calcular el Campo Electrico\n")
                    print ("\n 2) Grafica de las Lineas de campo y Equipotenciales de Cargas Puntuales\n")
                    op = int(input("\n Ingrese una Opcion : "))
                    os.system ("cls")

                if op == 1 :

                    Campo_Electrico ()

                else :

                    print ("\n\t\t\tGrafica de las Lineas de campo y Equipotenciales de Cargas Puntuales\n")
                    Lineas_Campo ()


                resp = input("\n\n ¿Desea Obtener Obtener otro Dato [Si-No]? : ")

        if numero == 2 :

            resp = "Si"

            while (resp == "Si" or resp == "si" or resp == "s" or resp == "S") :

                Minimos_Cuadrados ()
                resp = input("\n\n ¿Desea Obtener Obtener otro Dato [Si-No]? : ")
                os.system ("cls")





        if numero == 3 :

            resp = "Si"
            os.system ("cls")

            while (resp == "Si" or resp == "si" or resp == "s" or resp == "S") :

                Ley_de_Coulomb ()
                resp = input("\n\n ¿Desea Obtener Obtener otro Dato [Si-No]? : ")
                os.system ("cls")

        if numero == 4 :

            Ley_de_Ohm ()


        if numero == 5 :

            resp = "Si"
            os.system ("cls")

            while (resp == "Si" or resp == "si" or resp == "s" or resp == "S") :

                Codigo_de_Colores ()
                resp = input("\n\n ¿Desea Obtener Obtener otro Dato [Si-No]? : ")
                os.system ("cls")

        if numero == 6 :

            resp = "Si"
            os.system ("cls")
            while (resp == "Si" or resp == "si" or resp == "s" or resp == "S") :

                Calculadora_Resistencias ()
                resp = input("\n\n ¿Desea Obtener Obtener otro Dato [Si-No]? : ")
                os.system ("cls")

        if numero == 7 :

            resp = "Si"
            os.system ("cls")
            while (resp == "Si" or resp == "si" or resp == "s" or resp == "S") :

                Calculadora_Capacitores ()
                resp = input("\n\n ¿Desea Obtener Obtener otro Dato [Si-No]? : ")
                os.system ("cls")

        if numero == 8 :

            resp = "Si"
            os.system ("cls")
            while (resp == "Si" or resp == "si" or resp == "s" or resp == "S") :

                Error_Relativo ()
                resp = input("\n\n ¿Desea Obtener Obtener otro Dato [Si-No]? : ")
                os.system ("cls")

        if numero == 9 :

            resp = "Si"
            os.system ("cls")
            while (resp == "Si" or resp == "si" or resp == "s" or resp == "S") :

                Biot_Savart ()
                resp = input("\n\n ¿Desea Obtener Obtener otro Dato [Si-No]? : ")
                os.system ("cls")

        if numero == 10 :

            resp = "Si"
            os.system ("cls")
            while (resp == "Si" or resp == "si" or resp == "s" or resp == "S") :

                Grafica_RC ()
                resp = input("\n\n ¿Desea Obtener Obtener otro Dato [Si-No]? : ")
                os.system ("cls")


        os.system ("cls")
        respuesta = input("\n ¿Desea resolver otro problema [Si/No] : ")
        os.system ("cls")


Menu_Principal ()

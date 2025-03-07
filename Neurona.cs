using System;
using System.Collections.Generic;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase Neuronas para manejar las neuronas del PML de cada capa.
	/// </summary>
	public class Neurona
	{
		public string Nombre { get; set; }
		public double[] Pesos { get; set; } 
		public double Bias { get; set; }   
		public double Delta { get; set; } 
		public double Salida { get; set; }
		public List<Neurona> NeuronasSiguiente;
		/// <summary>
		/// Constructor de la clase Neurona.
		/// </summary>
		/// <param name="nombre">Nombre de la neurona.</param>
		/// <param name="numeroPesos">Número de pesos (conexiones con la capa anterior).</param>
		public Neurona(string nombre, int numeroPesos, int numeroNeuronasSiguientes)
		{
			Nombre = nombre;

			if (numeroPesos > 0)
			{
				Pesos = new double[numeroPesos];
				Random rand = new Random();
				for (int w = 0; w < numeroPesos; w++)
				{
					Pesos[w] = rand.NextDouble(); 
				}
				Bias = rand.NextDouble(); 
			}
			else
			{
				Pesos = null;
				Bias = 0;
			}
			Delta = 0;
			Salida = 0;
		}
	}
}

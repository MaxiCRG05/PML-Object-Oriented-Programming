using System;

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

		/// <summary>
		/// Constructor de la clase Neurona.
		/// </summary>
		/// <param name="nombre">Nombre de la neurona.</param>
		/// <param name="numeroPesos">Número de pesos (conexiones con la capa anterior).</param>
		public Neurona(string nombre, int numeroPesos)
		{
			Nombre = nombre;
			if (numeroPesos > 0)
			{
				Pesos = new double[numeroPesos];
				Random rand = new Random();
				for (int w = 0; w < numeroPesos; w++)
				{
					Pesos[w] = (rand.NextDouble() - 0.5) * 0.02; 
				}
				Bias = (rand.NextDouble() - 0.5) * 0.02; 
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

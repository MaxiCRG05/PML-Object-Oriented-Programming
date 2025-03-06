using System;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase Neuronas para manejar las neuronas del PML de cada capa.
	/// </summary>
	public class Neurona
	{
		public string Nombre { get; set; }
		public double[] Pesos { get; set; } // Pesos de la neurona
		public double Bias { get; set; }    // Sesgo de la neurona
		public double Delta { get; set; }   // Delta (error) de la neurona
		public double Salida { get; set; }  // Salida de la neurona

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

		/// <summary>
		/// Método para actualizar los pesos de la neurona durante la retropropagación.
		/// </summary>
		/// <param name="tasaAprendizaje">Tasa de aprendizaje.</param>
		/// <param name="entradas">Entradas de la	capa anterior.</param>
		public void ActualizarPesos(double tasaAprendizaje)
		{
			if (Pesos != null)
			{
				for (int w = 0; w < Pesos.Length; w++)
				{
					Pesos[w] += tasaAprendizaje * Delta * Salida;
				}
				Bias += tasaAprendizaje * Delta;
			}
		}
	}
}

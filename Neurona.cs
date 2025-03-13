using System;
using System.Collections.Generic;
using System.Drawing;
using static Perceptron_Multicapa_Colores.Capa;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase Neuronas para manejar las neuronas del PML de cada capa.
	/// </summary>
	public class Neurona
	{

		/// <summary>
		/// Generador de números.
		/// </summary>
		readonly Random rand = new Random();

		/// <summary>
		/// Nombre de la neurona.
		/// </summary>
		public String Nombre { get; set; }

		/// <summary>
		/// Pesos de la neurona dependiendo de las entradas que reciba.
		/// </summary>
		public double[] Pesos { get; set; }

		/// <summary>
		/// Bias de la neurona.
		/// </summary>
		public double Bias { get; set; }

		/// <summary>
		/// Delta (error) de la neurona.
		/// </summary>
		public double Delta { get; set; }

		/// <summary>
		/// Salida (predicción) de la neurona.
		/// </summary>
		public double Salida { get; set; }

		/// <summary>
		/// Constructor de la clase Neurona.
		/// </summary>
		/// <param name="nombre">Nombre de la neurona.</param>
		/// <param name="numeroPesos">Número de pesos (conexiones con la capa anterior).</param>
		public Neurona(String name, int numeroNeuronas, int numeroNeuronasCapaSiguiente, TipoCapa tipoCapa)
		{
			Nombre = name;

			if (tipoCapa == TipoCapa.Entrada)
			{
				Pesos = new double[numeroNeuronasCapaSiguiente];

				for (int i = 0; i < numeroNeuronasCapaSiguiente; i++)
					Pesos[i] = rand.NextDouble();
			}
			else if (tipoCapa == TipoCapa.Oculta)
			{
				Pesos = new double[numeroNeuronas];

				for (int i = 0; i < numeroNeuronas; i++)
					Pesos[i] = rand.NextDouble();
				IniciarBias();
			}
			else if (tipoCapa == TipoCapa.Salida)
			{
				Pesos = new double[numeroNeuronas];

				for (int i = 0; i < numeroNeuronas; i++)
					Pesos[i] = rand.NextDouble();
				IniciarBias();
			}
		}

		private void IniciarBias()
		{
			Bias = (double)new Random().NextDouble();
		}
	}
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Perceptron_Multicapa_Colores
{
	/// <summary>
	/// Clase Archivos para manejar los archivos para escribir y leer.
	/// </summary>
	class Archivos
	{
		/// <summary>
		/// Variables que se utilizaran para identificar dónde guardar el archivo.
		/// </summary>
		private readonly string ruta;

		/// <summary>
		/// StreamWriter para escribir en el archivo.
		/// </summary>
		private StreamWriter sw;

		/// <summary>
		/// StreamReader para leer el archivo.
		/// </summary>
		private StreamReader sr;

		/// <summary>
		/// Constructor de la clase Archivo.
		/// </summary>
		/// <param name="root">Variable para obtener la raiz para almacenar los archivos.</param>
		public Archivos(string root)
		{
			ruta = root;
		}

		/// <summary>
		/// Método para buscar un archivo, si lo encuentra pregunta si lo quiere limpiar, sino lo deja intacto; si no lo encuentra, crea el archivo.
		/// </summary>
		/// <param name="nombreArchivo">Variable para obtener el nombre del archivo para crearlo.</param>
		/// <returns>Verdadero: Si encontró el archivo. Falso: Si no lo encontró.</returns>
		public bool BuscarArchivo(string nombreArchivo)
		{
			if (File.Exists(ruta + nombreArchivo))
			{
				DialogResult dialogResult = MessageBox.Show($"El archivo {nombreArchivo} si se encuentra. ¿Deseas limpiarlo?", caption: $"Archivo {nombreArchivo}", buttons: MessageBoxButtons.YesNo);

				if (dialogResult == DialogResult.Yes)
				{
					CrearArchivo(nombreArchivo);
				}
				return true;
			}
			else
			{
				DialogResult dialogResult = MessageBox.Show("No se ha encontrado el archivo, ¿Deseas crearlo?", caption: $"Crear archivo {nombreArchivo}", buttons: MessageBoxButtons.YesNo);

				if (dialogResult == DialogResult.Yes)
				{
					CrearArchivo(nombreArchivo);
				}
				return false;
			}
		}

		/// <summary>
		/// Método para crear un archivo.
		/// </summary>
		/// <param name="nombreArchivo">Nombre del archivo para crearlo.</param>
		public void CrearArchivo(string nombreArchivo)
		{
			try
			{
				sw = new StreamWriter(ruta + nombreArchivo);
				sw.Close();
				MessageBox.Show("Se ha creado/limpiado correctamente el archivo.", caption: $"Creación del archivo: {nombreArchivo}");
			}
			catch (Exception e)
			{
				MessageBox.Show($"No se ha podido crear el archivo debido a un error.\nError:\n{e.Message}", caption: $"Creación del archivo: {nombreArchivo}");
				Console.WriteLine($"Excepcion: {e.Message}");
			}
		}

		/// <summary>
		/// Método para escribir en un archivo.
		/// </summary>
		/// <param name="texto">Texto el cual se escribirá en el archivo.</param>
		/// <param name="nombreArchivo">Nombre del archivo al cual se escribirá el texto.</param>
		public void EscribirArchivo(string texto, string nombreArchivo)
		{
			try
			{
				sw = new StreamWriter(ruta + nombreArchivo, true);
				sw.WriteLine(texto);
				sw.Close();
			}
			catch(Exception e)
			{
				MessageBox.Show($"No se ha podido escribir en el archivo debido a un error.\nError:\n{e.Message}", caption: $"Archivo {nombreArchivo}");
				Console.WriteLine($"Excepcion: {e.Message}");
			}
		}

		/// <summary>
		/// Método para leer una linea de un archivo.
		/// </summary>
		/// <param name="nombreArchivo">Nombre del archivo en el que se va a escribir.</param>
		/// <returns>Regresa el texto btenido de una linea.</returns>
		public string LeerArchivo(string nombreArchivo)
		{
			string texto;
			try
			{
				sr = new StreamReader(ruta + nombreArchivo);
				texto = sr.ReadLine();
				sr.Close();
				return texto;
			}
			catch(Exception e)
			{
				MessageBox.Show($"No se ha podido leer el archivo debido a un error.\n Error:\n {e.Message}", caption: $"Archivo: {nombreArchivo}");
				Console.WriteLine($"Excepcion: {e.Message}");
				return "";
			}
		}
	}
}
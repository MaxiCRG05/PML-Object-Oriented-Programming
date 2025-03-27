using System;
using System.Drawing;
using System.Linq;
using System.IO.Ports;
using System.Windows.Forms;
using System.IO;

namespace Perceptron_Multicapa_Colores
{
    /// <summary>
    /// Clase principal del formulario.
    /// </summary>
    public partial class Form1 : Form
    {
		SerialPort ArduinoPort = new SerialPort();

		/// <summary>
		/// Instancia de la clase Archivos para manejar los archivos.
		/// </summary>
		readonly Archivos archivos;

		/// <summary>
		/// Instancia de la clase PML para manejar el perceptrón multicapa.
		/// </summary>
		readonly PML pml;

		/// <summary>
		/// Color seleccionado.
		/// </summary>
		public Color color;

		/// <summary>
		/// Variable que sirve para saber si el perceptrón ha sido entrenado.
		/// </summary>
		private bool entrenado = false;

		/// <summary>
		/// Constructor de la clase Form1.
		/// </summary>
		public Form1()
        {
            pml = new PML(VariablesGlobales.n);

            archivos = new Archivos(VariablesGlobales.Ruta);
            archivos.BuscarArchivo(VariablesGlobales.Datos + VariablesGlobales.FormatoArchivos);

            InitializeComponent();

			ArduinoPort.PortName = "COM9";
			ArduinoPort.BaudRate = 9600;

			//if (archivos.BuscarArchivo(VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos))
   //         {
   //             DialogResult dialog = MessageBox.Show($"El archivo, si se ha encontrado. ¿Deseas cargar los pesos?", caption: "Perceptron", buttons: MessageBoxButtons.YesNo);
   //             if (dialog == DialogResult.Yes)
   //             {
   //                 //pml.CargarDatos();

   //                 btnProbar.Enabled = true;
   //                 button2.Enabled = false;
   //             }
   //         }
   //         else
   //         {
   //             btnProbar.Enabled = false; 
   //             button2.Enabled = true;    
   //         }
        }

        private void ActivarLed()
        {
            try
            {
                ArduinoPort.Open();

                ArduinoPort.Write(color.ToString());

			}
            catch(IOException e)
            {
                Console.WriteLine($"Error: {e.Message}");
            }
            finally
            {
                if (ArduinoPort.IsOpen)
                    ArduinoPort.Close();
            }
		}


		private void Probar()
        {
			double[] entradas = { color.R, color.G, color.B };

			double[] salida = pml.propagacion(entradas);

			int clasePredicha = Array.IndexOf(salida, salida.Max());

			string nombreColor = VariablesGlobales.NombresColores[clasePredicha];

			label2.Text = $"{nombreColor}";
			archivos.EscribirArchivo($"[{color.R}, {color.G}, {color.B}]\t Predicción: {nombreColor}", VariablesGlobales.Datos + VariablesGlobales.FormatoArchivos);

            MessageBox.Show($"Predicción: {nombreColor}", "Predicción");
        }

		/// <summary>
		/// Método para probar el perceptrón multicapa y poder predecir el resultado.
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void btnProbar_Click(object sender, EventArgs e)
        {
			if (entrenado)
			{
				Probar();
			}
		}

		/// <summary>
		/// Método para guardar los pesos del perceptrón multicapa.
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void btnGuardar_MouseClick(object sender, MouseEventArgs e)
        {
            //pml.GuardarDatos();  
        }

		/// <summary>
		/// Método para limpiar los pesos del archivo de configuración del perceptrón multicapa.
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void btnLimpiarPesos_MouseClick(object sender, MouseEventArgs e)
        {
            archivos.CrearArchivo(VariablesGlobales.Configuracion + VariablesGlobales.FormatoArchivos);
        }

		/// <summary>
		/// Método para cargar la imagen 1 y colocarla en el pictureBox1.
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void btnImagen1_MouseClick(object sender, MouseEventArgs e)
        {
            try
            {
                pictureBox1.Image = Properties.Resources.color;
                MessageBox.Show("Imagen 1 colocada con exito.", "Imagen");
            }
            catch
            {
                MessageBox.Show("La imagen 1 no se ha podido colocar.", "Imagen");
            }
        }

		/// <summary>
		/// Método para cargar la imagen 2 y colocarla en el pictureBox1.
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void btnImagen2_MouseClick_1(object sender, MouseEventArgs e)
        {
            try
            {
                pictureBox1.Image = Properties.Resources.color1;
                MessageBox.Show("Imagen 2 colocada con exito.", "Imagen");
            }
            catch
            {
                MessageBox.Show("La imagen 2 no se ha podido colocar.", "Imagen");
            }
        }

		/// <summary>
		/// Método para cargar una imagen personalizada y colocarla en el pictureBox1.
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void btnImagenPersonalizada_MouseClick(object sender, MouseEventArgs e)
        {
            try
            {

                openFileDialog1.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.gif";
                openFileDialog1.Title = "Seleccionar una imagen";

                if (openFileDialog1.ShowDialog() == DialogResult.OK)
                {
                    string rutaImagen = openFileDialog1.FileName;
                    pictureBox1.Image = Image.FromFile(rutaImagen);
                    MessageBox.Show("Imagen cargada con éxito.", "Imagen");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error al cargar la imagen: " + ex.Message, "Error");
            }
        }

		/// <summary>
		/// Método para entrenar el perceptrón multicapa.
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void button2_MouseClick(object sender, MouseEventArgs e)
        {
            button2.Enabled = false;
            pml.entrenar();
            btnProbar.Enabled = true;
            entrenado = true;
        }

		/// <summary>
		/// Método para obtener el color del pixel seleccionado en el pictureBox1.
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            if (pictureBox1.Image != null)
            {
                Bitmap bitmap = (Bitmap)pictureBox1.Image;
                int x = e.X * bitmap.Width / pictureBox1.ClientSize.Width;
                int y = e.Y * bitmap.Height / pictureBox1.ClientSize.Height;
                color = bitmap.GetPixel(x, y);
                registroColores.Text += $"R: {color.R} \tG: {color.G} \tB: {color.B}\n";
				//ActivarLed();
			}
            else
            {
                MessageBox.Show("No hay imagen cargada en el PictureBox.");
            }
        }

		/// <summary>
		/// Método para limpiar el registro de colores y el label2.
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void button1_MouseClick(object sender, MouseEventArgs e)
        {
            registroColores.Text = "";
            label2.Text = "";
        }

		private void pictureBox1_MouseDoubleClick(object sender, MouseEventArgs e)
		{
            if(entrenado)
            {
				Probar();
			}
		}
	}
}

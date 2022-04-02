package com.example.chessrefereeapp

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.*
import android.icu.text.SimpleDateFormat
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.ImageCapture.OnImageCapturedCallback
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import java.io.OutputStream
import java.util.*
import java.util.concurrent.Executor


private const val PERMISSION_REQUEST_CODE = 200

class GameActivity : AppCompatActivity(), View.OnClickListener, ImageAnalysis.Analyzer {

    private var pview: PreviewView? = null
    private var imview: ImageView? = null
    private var imageCapt: ImageCapture? = null
    private var analysis_on = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_game)

        if (! checkPermission()) {
            System.out.println("Nope nope")
            requestPermission();
        }

        var picture_bt = findViewById<Button>(R.id.picture_bt);
        var analysis_bt = findViewById<Button>(R.id.analysis_bt);
        pview = findViewById(R.id.previewView);
        imview = findViewById(R.id.imageView);

        picture_bt.setOnClickListener(this);
        analysis_bt.setOnClickListener(this);
        this.analysis_on = false;


        picture_bt.setOnClickListener(this)
        analysis_bt.setOnClickListener(this)
        analysis_on = false

        var provider = ProcessCameraProvider.getInstance(this)
        provider.addListener({
            try {
                val cameraProvider: ProcessCameraProvider = provider.get()
                startCamera(cameraProvider)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, getExecutor())
    }

    private fun checkPermission(): Boolean {
        return (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
        != PackageManager.PERMISSION_GRANTED)
    }

    private fun requestPermission() {
        ActivityCompat.requestPermissions(
            this, arrayOf(Manifest.permission.CAMERA),
            PERMISSION_REQUEST_CODE
        )
    }

    private fun startCamera(cameraProvider: ProcessCameraProvider) {
        cameraProvider.unbindAll()
        val camSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()
        var preview = Preview.Builder().build()
        preview.setSurfaceProvider(pview!!.surfaceProvider)
        imageCapt =
            ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()
        var imageAn =
            ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
        imageAn.setAnalyzer(getExecutor(), this)
        cameraProvider.bindToLifecycle(
            (this as LifecycleOwner),
            camSelector,
            preview,
            imageCapt,
            imageAn
        )
    }

    private fun getExecutor(): Executor {
        return ContextCompat.getMainExecutor(this)
    }
    override fun onClick(view: View) {
        when (view.id) {
            R.id.picture_bt -> capturePhoto()
            R.id.analysis_bt -> analysis_on = !analysis_on
        }
    }

    private fun capturePhoto() {
        //Es. SISDIG_2021127_189230.jpg
        val pictureName =
            "SISDIG_" + SimpleDateFormat("yyyyMMdd_HHmmss").format(Date()).toString() + ".jpeg"
        imageCapt!!.takePicture(
            getExecutor(),
            object : OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    //Create the picture's metadata
                    val newPictureDetails = ContentValues()
                    newPictureDetails.put(MediaStore.Images.Media._ID, pictureName)
                    newPictureDetails.put(
                        MediaStore.Images.Media.ORIENTATION,
                        image.imageInfo.rotationDegrees.toString()
                    )
                    newPictureDetails.put(MediaStore.Images.Media.DISPLAY_NAME, pictureName)
                    newPictureDetails.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    newPictureDetails.put(MediaStore.Images.Media.WIDTH, image.width)
                    newPictureDetails.put(MediaStore.Images.Media.HEIGHT, image.height)
                    newPictureDetails.put(
                        MediaStore.Images.Media.RELATIVE_PATH,
                        Environment.DIRECTORY_DCIM + "/SistemiDigitaliM"
                    )
                    var stream: OutputStream? = null
                    try {
                        //Add picture to MediaStore in order to make it accessible to other apps
                        //The result of the insert is the handle to the picture inside the MediaStore
                        var picturePublicUri = applicationContext.contentResolver.insert(
                            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                            newPictureDetails
                        )
                        if(picturePublicUri is Uri) {
                            stream =
                                applicationContext.contentResolver.openOutputStream(picturePublicUri)
                        }
                        val bitmapImage = pview!!.bitmap
                        if (!bitmapImage!!.compress(
                                Bitmap.CompressFormat.JPEG,
                                100,
                                stream
                            )
                        ) { //Save the image in the gallery
                            //Error
                        }
                        image.close()
                        stream?.close()
                        Toast.makeText(applicationContext, "Picture Taken", Toast.LENGTH_SHORT)
                            .show()
                    } catch (exception: java.lang.Exception) {
                        exception.printStackTrace()
                        Toast.makeText(
                            applicationContext,
                            "Error saving photo: " + exception.message,
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }
        )
    }

    override fun analyze(image: ImageProxy) {
        if (analysis_on) {
            var conv = pview!!.bitmap
            // Do something here!!!
            conv = conv?.let { toGrayscale(it) }
            imview!!.setImageBitmap(conv)
        }
        image.close()
    }

    fun toGrayscale(bmpOriginal: Bitmap): Bitmap? {
        val width: Int
        val height: Int
        height = bmpOriginal.height
        width = bmpOriginal.width
        val bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmpGrayscale)
        val paint = Paint()
        val cm = ColorMatrix()
        cm.setSaturation(0f)
        val f = ColorMatrixColorFilter(cm)
        paint.setColorFilter(f)
        c.drawBitmap(bmpOriginal, 0.0F, 0.0F, paint)
        return bmpGrayscale
    }

}
package com.example.chessrefereeapp

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.*
import android.icu.text.SimpleDateFormat
import android.net.Uri
import android.os.Bundle
import android.os.CountDownTimer
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
import java.io.OutputStream
import java.util.*
import java.util.concurrent.Executor
import java.util.concurrent.TimeUnit


private const val PERMISSION_REQUEST_CODE = 200

class GameActivity : AppCompatActivity(), View.OnClickListener, ImageAnalysis.Analyzer {


    private var pview: PreviewView? = null
    private var imview: ImageView? = null
    private var imageCapt: ImageCapture? = null
    private var analysis_on = false
    var textViewBlack: TextView? = null
    var textViewWhite: TextView? = null
    lateinit var countdown_timer_Black: CountDownTimer
    lateinit var countdown_timer_White: CountDownTimer
    var isRunning: Boolean = false
    var time_in_milli_seconds = 0L



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_game)

        if (!checkPermission()) {
            System.out.println("Nope nope")
            requestPermission()
        }
        startTimer("White")
        var picture_bt = findViewById<Button>(R.id.picture_bt)
        var analysis_bt = findViewById<Button>(R.id.analysis_bt)
        pview = findViewById(R.id.previewView)
        imview = findViewById(R.id.imageView)

        picture_bt.setOnClickListener(this)
        analysis_bt.setOnClickListener(this)
        this.analysis_on = false


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
    private fun checkTokens(hhToken: String, mmToken: String, ssToken: String): Long {
        val hh= hhToken.toLong()
        val mm= mmToken.toLong()
        val ss= ssToken.toLong()
        var res=0L;
        if( hh in 0..23 && mm in 0..59 && ss in 0..59 ){
            val hhToMillis = TimeUnit.HOURS.toMillis(hh)
            val mmToMillis = TimeUnit.MINUTES.toMillis(mm)
            val ssToMillis = TimeUnit.SECONDS.toMillis(ss)
            res= hhToMillis + mmToMillis + ssToMillis;
        }else {
            Toast.makeText(this,"Wrong time value!", Toast.LENGTH_LONG).show()
            finish()
        }
        return res
    }

    /*private fun startTimerBlack(){
        val initValueTimer=intent.extras?.get("EditTextTime").toString()
        if( initValueTimer.isNotBlank()) {
            var tokens= initValueTimer.split(":")
            if(tokens[0].length == 2 && tokens[1].length == 2 && tokens[2].length == 2 ) {

                time_in_milli_seconds = checkTokens(tokens[0],tokens[1],tokens[2])

                textViewBlack = findViewById(R.id.textView_countdown_Black)
                countdown_timer_Black = object : CountDownTimer(time_in_milli_seconds, 1000) {

                    // Callback function, fired on regular interval
                    override fun onTick(millisUntilFinished: Long) {
                        time_in_milli_seconds = millisUntilFinished
                        updateTextUI("Black")

                    }

                    override fun onFinish() {
                        textViewBlack?.text = "STOP"
                    }
                }
                countdown_timer_Black.start()
                isRunning = true
            }else{
                Toast.makeText(this,"Error: Wrong time value --> pattern hh:mm:ss! ", Toast.LENGTH_LONG).show()
                finish()
            }
        }else{
            Toast.makeText(this,"Error: Please digit time value!", Toast.LENGTH_LONG).show()
            finish()
        }
    }*/
    /*private fun startTimerWhite(){
        val initValueTimer=intent.extras?.get("EditTextTime").toString()
        if( initValueTimer.isNotBlank()) {
            val tokens= initValueTimer.split(":")
            if(tokens[0].length == 2 && tokens[1].length == 2 && tokens[2].length == 2 ) {

                time_in_milli_seconds = checkTokens(tokens[0],tokens[1],tokens[2])

                textViewWhite = findViewById(R.id.textView_countdown_White)
                countdown_timer_White = object : CountDownTimer(time_in_milli_seconds, 1000) {

                    // Callback function, fired on regular interval
                    override fun onTick(millisUntilFinished: Long) {
                        time_in_milli_seconds = millisUntilFinished
                        updateTextUI("White")

                    }

                    override fun onFinish() {
                        textViewWhite?.setText("STOP")
                    }
                }
                countdown_timer_White.start()
                isRunning = true
            }else{
                Toast.makeText(this,"Error: Wrong time value --> pattern hh:mm:ss! ", Toast.LENGTH_LONG).show()
                finish()
                }
        }else{
            Toast.makeText(this,"Error: Please digit time value!", Toast.LENGTH_LONG).show()
            finish()
        }
    }*/
    private fun startTimer(player: String){
        val initValueTimer=intent.extras?.get("EditTextTime").toString()
        if( initValueTimer.isNotBlank()) {
            val tokens= initValueTimer.split(":")
            if(tokens[0].length == 2 && tokens[1].length == 2 && tokens[2].length == 2 ) {

                time_in_milli_seconds = checkTokens(tokens[0],tokens[1],tokens[2])
                when (player){
                    "White" -> {
                        textViewWhite = findViewById(R.id.textView_countdown_White)
                        countdown_timer_White = object : CountDownTimer(time_in_milli_seconds, 1000) {

                            // Callback function, fired on regular interval
                            override fun onTick(millisUntilFinished: Long) {
                                time_in_milli_seconds = millisUntilFinished
                                updateTextUI("White")

                            }

                            override fun onFinish() {
                                textViewWhite?.setText("STOP")
                            }
                        }
                        countdown_timer_White.start()
                        isRunning = true
                    }
                    "Black" -> {
                        textViewBlack = findViewById(R.id.textView_countdown_Black)
                        countdown_timer_Black = object : CountDownTimer(time_in_milli_seconds, 1000) {

                            // Callback function, fired on regular interval
                            override fun onTick(millisUntilFinished: Long) {
                                time_in_milli_seconds = millisUntilFinished
                                updateTextUI("Black")

                            }

                            override fun onFinish() {
                                textViewBlack?.setText("STOP")
                            }
                        }
                        countdown_timer_Black.start()
                        isRunning = true
                    }
                }

            }else{
                Toast.makeText(this,"Error: Wrong time value --> pattern hh:mm:ss! ", Toast.LENGTH_LONG).show()
                finish()
            }
        }else{
            Toast.makeText(this,"Error: Please digit time value!", Toast.LENGTH_LONG).show()
            finish()
        }
    }
    /*private fun pauseTimer() {
        countdown_timer.cancel()
        isRunning = false
        //todo

    }*/
    private fun updateTextUI(player: String) {

        val hours=(time_in_milli_seconds / 1000) /3600
        val minute = ((time_in_milli_seconds / 1000) % 3600)/ 60
        val seconds = (time_in_milli_seconds / 1000) % 60
        when (player){
            "Black" -> textViewBlack?.text = String.format("%02d:%02d:%02d",hours,minute,seconds)
            "White" -> textViewWhite?.text = String.format("%02d:%02d:%02d",hours,minute,seconds)
        }

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
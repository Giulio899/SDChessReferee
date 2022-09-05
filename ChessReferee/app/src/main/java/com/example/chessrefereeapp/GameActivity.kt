package com.example.chessrefereeapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.CountDownTimer
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import java.io.ByteArrayOutputStream
import java.util.*
import java.util.concurrent.Executor
import java.util.concurrent.TimeUnit


private const val PERMISSION_REQUEST_CODE = 200
private const val CALIBRATE_BOARD = 0
private const val CALIBRATE_PIECES = 1
private const val GAME_CHECKING = 2

private enum class Player(val player: String){
    WHITE("White"),BLACK("Black")
}

class GameActivity : AppCompatActivity()/*, View.OnClickListener*/{
    private var pview: PreviewView? = null
    private var imview: ImageView? = null
    private var imageCapt: ImageCapture? = null
    private var analysis_on = false
    var textViewBlack: TextView? = null
    var textViewWhite: TextView? = null
    var textViewResult: TextView? = null
    lateinit var countdown_timer_Black: CountDownTimer
    lateinit var countdown_timer_White: CountDownTimer
    var time_in_milli_seconds = 0L
    private var handDetector : HandDetector? = null
    private var startGameButton : Button? = null
    private var turn : Player= Player.WHITE
    private var py = Python.getInstance()
    private var pyobj = py.getModule("chessDetection").callAttr("ChessDetection")
    private var step= CALIBRATE_BOARD
    private var calibrateButton: Button?=null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_game)

        if (!checkPermission()) {
            System.out.println("Nope nope")
            requestPermission()
        }
        startTimer(turn)

        //var picture_bt = findViewById<Button>(R.id.picture_bt)


        pview = findViewById(R.id.previewView)

        //imview = findViewById(R.id.imageView)

        textViewBlack = findViewById(R.id.textView_countdown_Black)
        textViewWhite = findViewById(R.id.textView_countdown_White)
        textViewResult = findViewById(R.id.textView_result)
        //picture_bt.setOnClickListener(this)
        this.analysis_on = false



        var provider = ProcessCameraProvider.getInstance(this)
        provider.addListener({
            try {
                val cameraProvider: ProcessCameraProvider = provider.get()
                startCamera(cameraProvider)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, getExecutor())

        handDetector= HandDetector(this)

        startGameButton= findViewById(R.id.startGame)
        calibrateButton=findViewById(R.id.calibrate_bt)

        disableButton(startGameButton)

        calibrateButton!!.setOnClickListener {
            calibrate_board()
        }
        startGameButton!!.setOnClickListener {

            startGame()

        }


    }
    private fun sendBitmapToDetector(){

        handDetector!!.processBitmap(pview!!.bitmap)
    }

    private fun switchPlayer(){
        if(turn == Player.WHITE)
            turn= Player.BLACK
        else if(turn== Player.BLACK)
            turn = Player.WHITE

    }
    private fun disableButton(button: Button?) {
        button!!.isEnabled = false
        button.isClickable=false
        //calibrateButton?.setBackgroundColor(R.color.black)
        //calibrateButton?.setBackgroundColor(ContextCompat.getColor(textView.context, R.color.greyish))
    }
    private fun enableButton(button: Button?) {
        button!!.isEnabled = true
        button.isClickable=true
        //calibrateButton?.setBackgroundColor(R.color.black)
        //calibrateButton?.setBackgroundColor(ContextCompat.getColor(textView.context, R.color.greyish))
    }
    private fun calibrate_board(){
        if(calibrateBoard()){
            textViewResult?.text="Calibrazione scacchiera completata inserisci i pezzi e premi START  per iniziare la partita"
            disableButton(calibrateButton)
            enableButton(startGameButton)
            step= GAME_CHECKING

        }
        else{
            textViewResult?.text= "Errore durante la calibrazione: premi di nuovo CALIBRATE per riprovare"
        }
    }
    private fun startGame() {

        /*if(step == CALIBRATE_BOARD){
            if(calibrateBoard()){
                textViewResult?.setText("Calibrazione scacchiera completata: " +
                                        "inserisci i pezzi e premi di nuovo START per iniziare la partita")
                step= GAME_CHECKING
            }
            else{
                textViewResult?.setText("Errore durante la calibrazione: " +
                                        "premi di nuovo START per riprovare")
            }
        }*/
        /*else if(step == CALIBRATE_PIECES){

        }*/
        if(step == GAME_CHECKING){
            var end = false
            var first_calibration_pieces=false
            while (!end){ // parte la partita

                println("entro nel ciclo ciclo")
                if(!first_calibration_pieces) {
                    checkGame()
                    first_calibration_pieces = true
                }
                while (!handDetector!!.isMoveDetected())
                    sendBitmapToDetector()

                Thread.sleep(2000)
                //stoppare timer
                // codice per invocare la parte di python
                val result = checkGame()
                //cambiare se tutto va bene il timer
                // valutare result
                if (result == "Fine") { // stringa fine di python
                    end = true
                    textViewResult?.text= "FINE PARTITA"
                }
                //pop up che mostra la mossa
                Toast.makeText(this, "Mossa fatta $result", Toast.LENGTH_LONG).show()
                /*textViewResult?.setText("Mossa fatta: " +
                                        result)*/
                switchPlayer()
            }
        }
        /*var end = false


        if (calibrateBoard()) {
            while (!end){ // parte la partita

                println("entro nel ciclo ciclo")

                while (!handDetector!!.isMoveDetected())
                    sendBitmapToDetector()

                Thread.sleep(2000)

                // codice per invocare la parte di python
                val result = checkGame()

                // valutare result
                if (result == "Fine") { // stringa fine di python
                    end = true
                }

                switchPlayer()
            }
        }
        else {
            // errore
        }*/

    }

    private fun calibrateBoard(): Boolean {
        // calibrate board
        val currentBoardImage= pview!!.bitmap
        val convertedBoardImage= convertBitmap(currentBoardImage)
        val obj = pyobj.callAttr("calibrate_board", convertedBoardImage)
        return obj.toBoolean()
    }

    private fun checkGame(): String {
        val currentBoardImage= pview!!.bitmap
        val convertedBoardImage= convertBitmap(currentBoardImage)
        val obj = pyobj.callAttr("game_checking", convertedBoardImage, turn.player)
        return obj.toString()
    }

    private fun convertBitmap(inputBitmap: Bitmap?): String?{
        if (inputBitmap == null)
            return null
        val resizedBitmap = Bitmap.createScaledBitmap(inputBitmap,640,480,false)
        val bos=ByteArrayOutputStream()
        resizedBitmap.compress(Bitmap.CompressFormat.PNG,100,bos)
        val imageAsByteArray= bos.toByteArray()
        return Base64.getEncoder().encodeToString(imageAsByteArray)
    }



    /*fun toGrayscale(bmpOriginal: Bitmap): Bitmap? {
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
    }*/

    /*override fun analyze(image: ImageProxy) {

        if (analysis_on) {


                imview!!.setImageBitmap(bitmap)
        }
        image.close()
    }*/



    private fun checkTokens(hhToken: String, mmToken: String, ssToken: String): Long {
        val hh = hhToken.toLong()
        val mm = mmToken.toLong()
        val ss = ssToken.toLong()

        if (hh !in 0..23 || mm !in 0..59 || ss !in 0..59) {
            Toast.makeText(this, "Wrong time value!", Toast.LENGTH_LONG).show()
            finish()

        }
        val hhToMillis = TimeUnit.HOURS.toMillis(hh)
        val mmToMillis = TimeUnit.MINUTES.toMillis(mm)
        val ssToMillis = TimeUnit.SECONDS.toMillis(ss)
        return hhToMillis + mmToMillis + ssToMillis;
    }

    private fun startTimer(player: Player){
        val initValueTimer=intent.extras?.get("EditTextTime").toString()
        if(initValueTimer.isBlank()) {
            Toast.makeText(this,"Error: Please digit time value!", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        val tokens= initValueTimer.split(":")

        if (tokens.count() != 3) {
            Toast.makeText(this,"Error: Wrong time value --> pattern hh:mm:ss! ", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        if(tokens[0].length != 2 || tokens[1].length != 2 || tokens[2].length != 2 ) {
            Toast.makeText(this,"Error: Wrong time value --> pattern hh:mm:ss! ", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        time_in_milli_seconds = checkTokens(tokens[0],tokens[1],tokens[2])
        when (player){
            Player.WHITE -> {


                countdown_timer_White = object : CountDownTimer(time_in_milli_seconds, 1000) {

                    // Callback function, fired on regular interval
                    override fun onTick(millisUntilFinished: Long) {

                        time_in_milli_seconds = millisUntilFinished
                        updateTextUI(Player.WHITE)

                    }

                    override fun onFinish() {
                        textViewWhite?.text = "STOP"
                    }
                }
                countdown_timer_White.start()

            }
            Player.BLACK -> {


                countdown_timer_Black = object : CountDownTimer(time_in_milli_seconds, 1000) {

                    // Callback function, fired on regular interval
                    override fun onTick(millisUntilFinished: Long) {
                        time_in_milli_seconds = millisUntilFinished
                        updateTextUI(Player.BLACK)

                    }

                    override fun onFinish() {
                        textViewBlack?.text = "STOP"
                    }
                }
                countdown_timer_Black.start()

            }
        }




    }

    private fun pauseTimer(player: Player){
        when(player){
            Player.BLACK -> countdown_timer_White.cancel()
            Player.WHITE-> countdown_timer_Black.cancel()
        }

    }
    private fun updateTextUI(player: Player) {

        val hours=(time_in_milli_seconds / 1000) /3600
        val minute = ((time_in_milli_seconds / 1000) % 3600)/ 60
        val seconds = (time_in_milli_seconds / 1000) % 60
        when (player){
            Player.BLACK -> textViewBlack?.text = String.format("%02d:%02d:%02d",hours,minute,seconds)
            Player.WHITE -> textViewWhite?.text = String.format("%02d:%02d:%02d",hours,minute,seconds)
        }

    }

    private fun checkPermission(): Boolean {
        return (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
        != PackageManager.PERMISSION_GRANTED)
    }

    private fun requestPermission() {
        ActivityCompat.requestPermissions(
            this, arrayOf(Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE),
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
        /*var imageAn =
            ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
        imageAn.setAnalyzer(getExecutor(), this)*/
        cameraProvider.bindToLifecycle(
            (this as LifecycleOwner),
            camSelector,
            preview,
            imageCapt,
            //imageAn
        )
    }



    private fun getExecutor(): Executor {

        return ContextCompat.getMainExecutor(this)
    }
    /*override fun onClick(view: View) {
        when (view.id) {

            R.id.analysis_bt -> analysis_on = !analysis_on

        }
    }*/

    /*private fun capturePhoto() {
        //Es. SISDIG_2021127_189230.jpg
        val pictureName =
            "SISDIG_" + SimpleDateFormat("yyyyMMdd_HHmmss").format(Date()).toString() + ".jpeg"
        imageCapt!!.takePicture(
            getExecutor(),
            object : ImageCapture.OnImageCapturedCallback() {
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
                        val bitmapImage = getLastBitMap()
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
    }*/







}
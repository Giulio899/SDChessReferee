package com.example.chessrefereeapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.*
import android.view.View
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import com.example.chessrefereeapp.ui.controller.ChessBoardAdapter
import com.example.chessrefereeapp.ui.controller.ChessBoardController
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.example.chessrefereeapp.timer.CountDownTimerWithPause
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

class GameActivity : AppCompatActivity(),ImageAnalysis.Analyzer /*, View.OnClickListener*/{
    private var pview: PreviewView? = null
    private var imview: ImageView? = null
    private var imageCapt: ImageCapture? = null
    private var analysis_on = false
    var textViewBlack: TextView? = null
    var textViewWhite: TextView? = null
    var textViewResult: TextView? = null
    //lateinit var countdown_timer_Black: CountDownTimer
    //lateinit var countdown_timer_White: CountDownTimer
    //modifiche lorenzo
    lateinit var countdown_timer_Black: CountDownTimerWithPause
    lateinit var countdown_timer_White: CountDownTimerWithPause
    var remaining_time_white = 0L
    var remaining_time_black = 0L
    private lateinit var chessBoardView: GridView
    private var handDetector : HandDetector? = null
    private var startGameButton : Button? = null
    private var switch_bt : Button? = null
    private var turn : Player= Player.WHITE
    private var py = Python.getInstance()
    private var pyobj = py.getModule("chessDetection").callAttr("ChessDetection")
    private var step= CALIBRATE_BOARD
    private var calibrateButton: Button?=null
    private var controller: ChessBoardController? =null
    private var adapter: ChessBoardAdapter? =null
    private var currentBoardImage : Bitmap?=null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_game)

        if (!checkPermission()) {
            System.out.println("Nope nope")
            requestPermission()
        }

        initTimer()
        pview = findViewById<PreviewView>(R.id.previewView)
        chessBoardView = findViewById(R.id.chessBoard)
        textViewBlack = findViewById(R.id.textView_countdown_Black)
        textViewWhite = findViewById(R.id.textView_countdown_White)
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
        switch_bt=findViewById<Button>(R.id.switch_button)
        disableButton(startGameButton)
        controller= ChessBoardController()
        adapter = ChessBoardAdapter(this, controller!!.get_board_as_List())
        chessBoardView.adapter = adapter


        calibrateButton!!.setOnClickListener {
            calibrate_board()
        }
        startGameButton!!.setOnClickListener {
            Thread(Runnable {
                startGame()
            }).start()
        }
        switch_bt!!.setOnClickListener {
            if(chessBoardView.visibility== View.VISIBLE){
                chessBoardView.visibility= View.INVISIBLE
                pview!!.visibility=View.VISIBLE
            }else if(pview!!.visibility==View.VISIBLE){
                chessBoardView.visibility= View.VISIBLE
                pview!!.visibility=View.INVISIBLE
            }

        }

    }

    private fun sendBitmapToDetector(bitmap: Bitmap){

        handDetector!!.processBitmap(bitmap)
    }
    private fun switchPlayer(){
        if(turn == Player.WHITE)
            turn = Player.BLACK
        else if(turn == Player.BLACK)
            turn = Player.WHITE

    }
    private fun disableButton(button: Button?) {
        button!!.isEnabled = false
        button.isClickable=false


    }
    private fun enableButton(button: Button?) {
        button!!.isEnabled = true
        button.isClickable=true


    }
    private fun calibrate_board(){

        if(calibrateBoard()){
            disableButton(calibrateButton)
            enableButton(startGameButton)
            step= GAME_CHECKING
            Toast.makeText(this,"Calibrazione scacchiera completata inserisci i pezzi e premi START  per iniziare la partita",Toast.LENGTH_LONG).show()
        }
        else{

            Toast.makeText(this,"Errore durante la calibrazione: premi di nuovo CALIBRATE per riprovare",Toast.LENGTH_LONG).show()

        }
        //mock behavior
        /*disableButton(calibrateButton)
        enableButton(startGameButton)
        step= GAME_CHECKING*/
    }


    private fun startGame() {

            startTimer()


            if (step == GAME_CHECKING) {

                var end = false
                var first_calibration_pieces = false
                //mock var index_move = 0
                analysis_on=!analysis_on
                // mock val moves = listOf<String>("d2/d4", "d7/d5", "f2/f4", "h7/h5","Fine"
                runOnUiThread { disableButton(startGameButton)}
                while (!end) { // parte la partita
                     println("entro nel ciclo ciclo")
                    if(!first_calibration_pieces) {
                        val first_res = checkGame()
                        analysis_on=!analysis_on
                        if (first_res == "START") {

                            runOnUiThread { Toast.makeText(
                                this,
                                "Calibrazione Pezzi Completata",
                                Toast.LENGTH_LONG
                            ).show()}

                            first_calibration_pieces = true
                            runOnUiThread { disableButton(startGameButton)}

                        }

                    }

                    while(!handDetector!!.isMoveDetected()){}

                    Thread.sleep(2000)
                    // mock var result = moves[index_move]
                    var result = checkGame()
                    analysis_on=!analysis_on
                    controller!!.move(result)
                    runOnUiThread { adapter!!.notifyDataSetChanged() }
                    controller!!.switchPlayer()
                    /*controller!!.checkLiveData().observe(this, androidx.lifecycle.Observer {
                        if (it == "CHECKMATE") {

                            AlertDialog.Builder(this).setMessage("CHECKMATE !!").show()
                            result = "Fine"
                        } else {

                            Toast.makeText(this, "CHECK!!", Toast.LENGTH_SHORT).show()

                        }
                    })*/
                    if (result == "Fine") { // stringa fine di python
                        end = true
                        runOnUiThread { AlertDialog.Builder(this).setMessage("FINE PARTITA").show()}
                        analysis_on=!analysis_on

                    }
                    switchPlayer()
                    switchTimer()
                    //mock index_move++

                }

            }


    }

    private fun calibrateBoard(): Boolean {
        // calibrate board
        val currentBoardImage= pview!!.bitmap
        val convertedBoardImage= convertBitmap(currentBoardImage)
        val obj = pyobj.callAttr("calibrate_board", convertedBoardImage)
        return obj.toBoolean()
    }

    private fun checkGame(): String {
        analysis_on=!analysis_on
        //val currentBoardImage= pview!!.bitmap
        runOnUiThread { currentBoardImage=pview!!.bitmap }
        Thread.sleep(1000)
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

    override fun analyze(image: ImageProxy) {

        if (analysis_on) {

            var curr_bitmap= pview!!.bitmap
            runOnUiThread {
               sendBitmapToDetector(curr_bitmap!!)
            }

        }
        image.close()
    }


    private fun initTimer() {
        // get time from input
        val initValueTimer=intent.extras?.get("EditTextTime").toString()
        val mm = initValueTimer.toLong()
        val mmToMillis = TimeUnit.MINUTES.toMillis(mm)
        remaining_time_black = mmToMillis
        remaining_time_white = mmToMillis
        updateTextUI(Player.WHITE)
        updateTextUI(Player.BLACK)

        countdown_timer_White = object : CountDownTimerWithPause(remaining_time_white, 1000, false) {

            override fun onTick(millisUntilFinished: Long) {
                remaining_time_white = millisUntilFinished
                updateTextUI(Player.WHITE)
            }

            override fun onFinish() {
                textViewWhite?.text = "STOP"
            }
        }

        countdown_timer_Black = object : CountDownTimerWithPause(remaining_time_black, 1000, false) {
            override fun onTick(millisUntilFinished: Long) {
                remaining_time_black = millisUntilFinished
                updateTextUI(Player.BLACK)

            }

            override fun onFinish() {
                textViewBlack?.text = "STOP"
            }
        }
        countdown_timer_White.create()
        countdown_timer_Black.create()


    }

    private fun switchTimer() {

        pauseTimer()
        startTimer()
    }

    private fun startTimer(){

        when (turn){
            Player.WHITE -> {

                countdown_timer_White.resume()
            }
            Player.BLACK -> {

                countdown_timer_Black.resume()


            }
        }

    }

    private fun pauseTimer(){
        when(turn){
            Player.BLACK -> countdown_timer_White.pause()
            Player.WHITE-> countdown_timer_Black.pause()
        }

    }
    private fun updateTextUI(player: Player) {

        when (player){
            Player.BLACK -> {
                val hours=(remaining_time_black / 1000) /3600
                val minute = ((remaining_time_black / 1000) % 3600)/ 60
                val seconds = (remaining_time_black / 1000) % 60
                textViewBlack?.text = String.format("%02d:%02d:%02d",hours,minute,seconds)
            }
            Player.WHITE -> {
                val hours=(remaining_time_white / 1000) /3600
                val minute = ((remaining_time_white / 1000) % 3600)/ 60
                val seconds = (remaining_time_white / 1000) % 60
                textViewWhite?.text = String.format("%02d:%02d:%02d",hours,minute,seconds)
            }
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
        /*imageCapt =
            ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()*/
        var imageAn =
            ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
        imageAn.setAnalyzer(getExecutor(), this)
        cameraProvider.bindToLifecycle(
            (this as LifecycleOwner),
            camSelector,
            preview,
            //imageCapt,
            imageAn
        )
    }



    private fun getExecutor(): Executor {

        return ContextCompat.getMainExecutor(this)
    }










}
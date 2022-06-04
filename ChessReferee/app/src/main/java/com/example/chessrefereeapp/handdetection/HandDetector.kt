package com.example.chessrefereeapp.handdetection

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.SurfaceTexture
import android.view.View
import android.widget.FrameLayout
import androidx.camera.core.CameraProvider
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.drawToBitmap
import com.google.mediapipe.components.FrameProcessor
import com.google.mediapipe.framework.TextureFrame
import com.google.mediapipe.solutioncore.CameraInput
import com.google.mediapipe.solutioncore.SolutionGlSurfaceView
import com.google.mediapipe.solutions.hands.Hands
import com.google.mediapipe.solutions.hands.HandsOptions
import com.google.mediapipe.solutions.hands.HandsResult
import java.util.concurrent.Executor

class HandDetector(private var activity: Activity?, private var frameLayout: FrameLayout?){

    private var cameraInput: CameraInput? = CameraInput(activity)
    private var hands: Hands? = Hands(activity,
        HandsOptions.builder()
            .setStaticImageMode(false)
            .setMaxNumHands(2)
            .setRunOnGpu(true)
            .build())

    private var glSurfaceView: SolutionGlSurfaceView<HandsResult>? = SolutionGlSurfaceView(activity, hands!!.glContext, hands!!.glMajorVersion)
    private var handDetected: Boolean = false

    private var moveDetected: Boolean = false


    init {



        requireNotNull(activity)
        requireNotNull(frameLayout)
        setResultListenerHand()
        setFrameListenerCamera()
        initializeRenderer()
        glSurfaceView!!.post { startCamera() }

        updateLayout()

    }



    fun getMoveDetected(): Boolean {
        return moveDetected
    }

    private fun setResultListenerHand(){
        hands!!.setResultListener { handsResult: HandsResult ->

            if(handsResult.multiHandLandmarks().isNotEmpty())  {
                println("VEDO LA MANO")

                this.handDetected = true
            }else if(handDetected){

                handDetected = false
                moveDetected= true


            }else{

                moveDetected= false
            }

            glSurfaceView!!.setRenderData(handsResult)
            glSurfaceView!!.requestRender()
        }

    }


    private fun startCamera() {

        cameraInput!!.start(
            activity,
            hands!!.glContext,
            CameraInput.CameraFacing.BACK,
            glSurfaceView!!.width,
            glSurfaceView!!.height
        )
    }

    private fun initializeRenderer(){
        glSurfaceView!!.setSolutionResultRenderer(HandsResultGlRenderer())
        glSurfaceView!!.setRenderInputImage(true)
    }



    private fun setFrameListenerCamera(){
        cameraInput!!.setNewFrameListener { textureFrame: TextureFrame? ->

            hands!!.send(
                textureFrame
            )
        }


    }

    private fun updateLayout() {
        frameLayout?.removeAllViewsInLayout()
        frameLayout?.addView(glSurfaceView)
        glSurfaceView!!.visibility = View.VISIBLE
        frameLayout?.requestLayout()
    }

}

package com.example.chessrefereeapp

import android.app.Activity
import android.graphics.Bitmap
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.google.mediapipe.solutions.hands.Hands
import com.google.mediapipe.solutions.hands.HandsOptions
import com.google.mediapipe.solutions.hands.HandsResult


class HandDetector(private var activity: Activity?) {

    private var hands: Hands? = Hands(
        activity,
        HandsOptions.builder()
            .setStaticImageMode(true)
            .setMaxNumHands(2)
            .setMinDetectionConfidence(0.5F)
            .setMinTrackingConfidence(0.5F)
            .setRunOnGpu(true)
            .build()
    )

    private var handDetected :Boolean=false
    private var currentHandsResult: HandsResult? = null




    init {
        requireNotNull(activity)
        setResultListenerHand()

    }



    fun processBitmap(inputBitmap: Bitmap?) {

        hands!!.send(inputBitmap)

    }


    private fun setResultListenerHand() {

            hands!!.setResultListener { handsResult: HandsResult ->

                currentHandsResult = handsResult
            }


    }

    fun isMoveDetected(): Boolean {

        if(this.currentHandsResult != null && this.currentHandsResult!!.multiHandLandmarks().isNotEmpty()){
            println("VEDO LA MANOOOOOO")
            handDetected=true

        }else if (handDetected && this.currentHandsResult!!.multiHandLandmarks().isEmpty()) {

            handDetected = false
            return true
        }

        return false

    }
}
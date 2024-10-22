package com.example.chessrefereeapp

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.EditText
import android.widget.NumberPicker
import android.widget.TextView
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : AppCompatActivity() {


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val numberPicker = findViewById<NumberPicker>(R.id.timePicker)
        numberPicker.maxValue = 120
        numberPicker.minValue = 0
        numberPicker.value = 30


        if(!Python.isStarted()){
            Python.start( AndroidPlatform(this))
        }

    }

    fun startGame(view: View) {
        val intent = Intent(this, GameActivity::class.java)
        val editTextTime=findViewById<NumberPicker>(R.id.timePicker)
        intent.putExtra("EditTextTime",editTextTime.value.toString())
        startActivity(intent)
    }
}
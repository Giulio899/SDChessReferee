package com.example.chessrefereeapp

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.EditText

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }

    fun startGame(view: View) {
        val intent = Intent(this, GameActivity::class.java)
        var editTextTime=findViewById<EditText>(R.id.editTextTime);
        intent.putExtra("EditTextTime",editTextTime.text.toString());
        startActivity(intent)
    }
}
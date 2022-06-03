package com.example.chessrefereeapp

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.EditText
import android.widget.TextView
import com.chaquo.python.Python

class MainActivity : AppCompatActivity() {


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        val py = Python.getInstance()
        val pyobj = py.getModule("prova").callAttr("Counter")
        val obj = pyobj.callAttr("get")
        val obj2 = pyobj.callAttr("plus")
        val obj3 = pyobj.callAttr("get")

        val text = findViewById<TextView>(R.id.provapy)
        text.setText(obj.toString())

        val text2 = findViewById<TextView>(R.id.provapy2)
        text2.setText(obj3.toString())
    }

    fun startGame(view: View) {
        val intent = Intent(this, GameActivity::class.java)
        val editTextTime=findViewById<EditText>(R.id.editTextTime)
        intent.putExtra("EditTextTime",editTextTime.text.toString())
        startActivity(intent)
    }
}
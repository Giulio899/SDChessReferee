package com.example.chessrefereeapp.ui.controller

import android.content.Context
import android.graphics.Bitmap
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.ImageView
import android.widget.TextView
import com.example.chessrefereeapp.ui.Tile
import com.example.chessrefereeapp.R



class ChessBoardAdapter(context: Context, tileArrayList: List<Tile>?) : ArrayAdapter<Tile>(context, 0, tileArrayList!!) {

    private fun select_image(letter: String,position: Int): String{ // 0-9, 101,109
        if(letter[0].isUpperCase())
            return "w"+ letter.lowercase()
        if(letter in "abcdefgh" && (position in 0..9 || position in 90..99) )
            return letter
        if(letter[0].isDigit())
            return "c$letter"
        return "b$letter"
    }
    override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {

        var listitemView = convertView
        if (listitemView == null) {
            // Layout Inflater inflates each item to be displayed in GridView.
            listitemView = LayoutInflater.from(context).inflate(R.layout.card_item, parent, false)
        }

        val tile: Tile? = getItem(position)
        //val square = listitemView!!.findViewById<TextView>(R.id.square)
        val square = listitemView!!.findViewById<ImageView>(R.id.square)


        if (tile != null) {

            if(tile.piece_letter== " "){
                square.setImageResource(context.resources.getIdentifier("empty", "drawable", context.packageName))

            }else{

                square.setImageResource(
                    context.resources.getIdentifier(
                        select_image(tile.piece_letter,position)  ,
                        "drawable",
                        context.packageName
                    )
                )
            }


            square.setBackgroundColor(tile.color)
        }

        return listitemView
    }





}

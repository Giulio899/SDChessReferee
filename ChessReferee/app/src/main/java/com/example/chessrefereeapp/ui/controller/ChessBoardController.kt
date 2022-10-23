package com.example.chessrefereeapp.ui.controller

import android.graphics.Color
import com.example.chessrefereeapp.ui.Tile
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData

private enum class Player(val player: Int){
    WHITE(1),BLACK(8)
}
private enum class CastlingType(val type: String){
    LONG("LONG"),SHORT("SHORT")
}
class ChessBoardController {

    private var chessBoard= mutableListOf<MutableList<Tile>>()
    private val columns_letter= "abcdefgh"
    private var columns_map= mutableMapOf<String,Int>()
    private var turn : Player = Player.WHITE
    private val _checkLiveData = MutableLiveData<String>()
    private val border_color= "#FAF9F6"
    fun checkLiveData(): LiveData<String>{
        return _checkLiveData
    }

    private val castling_map= mapOf<CastlingType,String>(
        CastlingType.SHORT to "eghf",
        CastlingType.LONG to "ecad"
    )



    init {
        initChessBoard()
    }


    fun switchPlayer(){
        if(turn == Player.WHITE)
            turn= Player.BLACK
        else if(turn== Player.BLACK)
            turn = Player.WHITE

    }

    fun get_board_as_List():List<Tile> {
        val result_list= mutableListOf<Tile>()
        chessBoard.forEach {
            it.forEach{ piece->result_list.add(piece)}
        }
        return result_list
    }
    private fun addLetterColumnToView(){
        var temp=emptyList<Tile>().toMutableList()
        temp.add(Tile(Color.parseColor(border_color)," "))
        for(el in columns_letter)
            temp.add(Tile(Color.parseColor(border_color),el.toString()))
        temp.add(Tile(Color.parseColor(border_color)," "))
        chessBoard.add(temp)
    }
    private fun initChessBoard(){
        //var pieces_letter="rnbqkbnrpppppppp                                PPPPPPPPRNBGKBNR"
        var pieces_letter="RNBQKBNRPPPPPPPP                                pppppppprnbqkbnr"
        var temp_row= mutableListOf<Tile>()
        var cambio=true
        addLetterColumnToView()
        for (row in 0..7){
            temp_row.add(Tile(Color.parseColor(border_color),(row+1).toString()))
            for (col in 0..7){

                if (cambio) {
                    temp_row.add(Tile(Color.parseColor("#037d50"),pieces_letter[col].toString()))
                    //temp_row.add(pieces_letter[col].toString())
                    cambio = false

                } else {
                    temp_row.add(Tile(Color.parseColor(border_color),pieces_letter[col].toString()))
                    //temp_row.add(pieces_letter[col].toString())
                    cambio = true

                }

            }

            temp_row.add(Tile(Color.parseColor(border_color),(row+1).toString()))
            chessBoard.add(1,temp_row) // chessBoard.add(temp_row)
            pieces_letter = (pieces_letter.reversed()).dropLast(8).reversed()
            temp_row= emptyList<Tile>().toMutableList()
            cambio=!cambio

        }
        addLetterColumnToView()

        for(number in 0..7)
            columns_map[columns_letter[number].toString()] = number

    }


    fun pb(){


        chessBoard.forEach {
            it.forEach {
                piece-> (
                    print(piece.piece_letter)
                    )

            }
            println("\n")
        }

    }
    //arrocco bianco corto 0-0 e1/g1 --- h1/f1 *
    //arrocco nero lungo 0-0-0 e8/c8 --- a8/d8

    //arrocco nero corto 0-0-0 e8/g8 --- h8/f8 *
    //arrocco bianco lungo 0-0-0 e1/c1 --- a1/d1

    private fun get_tile_for_castilng(castlingType: CastlingType) :List<Tile>{
        val castlingType_values=castling_map.get(castlingType)
        var tile_list= mutableListOf<Tile>()
        castlingType_values!!.forEach { tile_list.add(
            getTileInPosition(it.toString()+ turn.player )
        ) }
        return tile_list

    }
    private fun castling(castlingType: CastlingType){ //short -long
        val tile_list= get_tile_for_castilng(castlingType) // list: king - blank king - rook - blank rook (blank <nome pezzo> indica la casella vuota dove il <nome pezzo> deve spostarsi)
        tile_list[1].piece_letter=tile_list[0].piece_letter
        tile_list[0].piece_letter=" "
        //switch rook
        tile_list[3].piece_letter=tile_list[2].piece_letter
        tile_list[2].piece_letter=" "





    }
    fun move(move:String){ // d2/d4
        println("MOSSA AL CONTROLLER : " +
                "$move")
        if(move== "0-0"){

            castling(CastlingType.SHORT)
            return
        }

        if(move== "0-0-0"){

            castling(CastlingType.LONG)
            return
        }

        //check formato mossa
        if(!move.matches(Regex("[a-z][0-9]/[a-z][0-9]([/]?[+,#,q,Q])?"))){
            println("mossa non valida")
            return
        }

        val move_tokens=move.split("/")
        val first_part=move_tokens[0] //d2
        val second_part=move_tokens[1] //d4
        var third_part=""
        if(move_tokens.count()==3)
            third_part=move_tokens[2] // q or Q or # or +

        val first_tile= getTileInPosition(first_part) // casella da dove parto
        val second_tile= getTileInPosition(second_part) // casella da dove parto


        // caso mossa con/senza cattura
        second_tile.piece_letter = if (third_part=="q" || third_part=="Q") third_part else first_tile.piece_letter
        first_tile.piece_letter=" "

        //check-checkmate
        /*if(move.contains("#")) {
            _checkLiveData.value = "CHECKMATE"
            return
        }
        if(move.contains("+")) {
            _checkLiveData.value = "CHECK"
            return
        }*/


    }


    private fun getTileInPosition(position_as_string :String): Tile{

        val column_as_number=findColumn(position_as_string)
        val row_as_number= findRow(position_as_string)
        val row_list=chessBoard.get(row_as_number)
        return row_list.get(column_as_number)


    }

    private fun findColumn(position_as_string :String) : Int{
        val column_as_letter=position_as_string[0]
        val column_as_number= columns_map.get(column_as_letter.toString())

        return column_as_number!! + 1
    }
    private fun findRow(position_as_string :String) : Int{
        val row_as_number=position_as_string[1].toString().toInt() //RICORDA: qui toglievo -1
        return 8-row_as_number +1 //(row_number)

    }




}
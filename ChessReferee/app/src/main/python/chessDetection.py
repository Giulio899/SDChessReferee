import pickle
import argparse
import cv2
import numpy as np
import chess
from chess import polyglot
#from chess import uci
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image
import base64
import io
CHESSBOARD_WIDTH = 8
SQUARE_WIDTH = 100
CHESSBOARD_SQUARES = 64

class ChessDetection():
    def __init__(self):
        self.board = chess.Board()
        self.board_matrix = None
        self.board_corners= None

    def calibrate_board(self, bmp):
        # 1st calibration image - blank board

        # live corner detection with preview
        # cap = cv2.VideoCapture(CAMERA_INDEX)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, CV_CAP_PROP_FRAME_WIDTH)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CV_CAP_PROP_FRAME_HEIGHT)
        # while True:
        #     ret, img = cap.read()
        #     _, img = get_chessboard_corners(img, draw_corners=True)
        #     cv2.imshow('Once the coloured lines appear - press y to confirm', img)
        #     if cv2.waitKey(1) & 0xFF == ord('y'):
        #         break
        # cap.release()
        # cv2.destroyAllWindows()

        # final image capture for calibration
        #img = get_single_image()
        #print('Camera resolution', img.shape[0:2])

        #decode from base64
        img = self.decodeImage(bmp)
        #cv2.imwrite('./calibration/empty-board.jpg', img)
        # final corner detection
        corners, calib_img1 = self.get_chessboard_corners(img)

        # check if valid then save
        if corners is None:
            return False
        elif corners.shape != (49, 1, 2):
            raise False
        else:
            # save corners
            #np.save('./calibration/corners.npy', corners)
            self.board_corners=corners
            # save projected blank board
            board_img = self.project_board(calib_img1, corners)
            #square_colours = get_board_colours(board_img)
            board_img = self.label_squares(board_img)
            #cv2.imwrite('./calibration/empty-board-projected.jpg', board_img)

            print('Calibration step 1 completed - corners detected and saved')
        return True

    def label_squares(self, board_img, recom_move=None, alpha=0.2):
        """
        Overlay chess coordinates onto board image
        """
        def sq_offset(sq, xr, yr):
            x = (sq % CHESSBOARD_WIDTH)
            y = CHESSBOARD_WIDTH - 1 - int(sq / CHESSBOARD_WIDTH)
            x = int((x + xr) * SQUARE_WIDTH)
            y = int((y + yr) * SQUARE_WIDTH)
            return x, y

        # draw square names
        overlay = board_img.copy()
        for sq in range(CHESSBOARD_SQUARES):
            x, y = sq_offset(sq, 0.1, 0.8)
            cv2.putText(overlay, chess.SQUARE_NAMES[sq], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 2)

        cv2.addWeighted(overlay, alpha, board_img, 1 - alpha, 0, board_img)

        # draw recommended move
        if recom_move:
            # square numbers
            from_sq = chess.SQUARE_NAMES.index(str(recom_move)[0:2])
            to_sq = chess.SQUARE_NAMES.index(str(recom_move)[2:4])

            # outline recommended move using line and 2 circles
            board_img = cv2.line(board_img, sq_offset(from_sq, 0.5, 0.5), sq_offset(to_sq, 0.5, 0.5), (0, 0, 255), 3)
            board_img = cv2.circle(board_img, sq_offset(from_sq, 0.5, 0.5), int(SQUARE_WIDTH / 2), (0, 0, 255), 5)
            board_img = cv2.circle(board_img, sq_offset(to_sq, 0.5, 0.5), int(SQUARE_WIDTH / 7), (0, 0, 255), -1)

        return board_img

    def project_board(self, img, corners):
        """
        Takes a raw image and projects the squares within the cropped board.
        Input:
            img: multi-channel image, 3D array
            corners: corners from get_chessboard_corners
        Output:
            board_img: RGB image
        """

        def impute_corners(corners):
            """
            Since findChessboardCorners only detects the inside corners, this function
            resizes (w+2, h+2) and imputes the extra corners.
            Assumes w == h.
            """
            w = CHESSBOARD_WIDTH - 1

            # new array
            new_cnr = np.zeros((w+2, w+2, 2), np.float32)

            # inside edges
            new_cnr[1:w+1, 1:w+1, :] = corners.reshape(w, w, 2)

            # outside edges
            new_cnr[0, :, :] = new_cnr[1, :, :] - (new_cnr[2, :, :] - new_cnr[1, :, :])
            new_cnr[w+1, :, :] = new_cnr[w, :, :] + (new_cnr[w, :, :] - new_cnr[w-1, :, :])
            new_cnr[:, 0, :] = new_cnr[:, 1, :] - (new_cnr[:, 2, :] - new_cnr[:, 1, :])
            new_cnr[:, w+1, :] = new_cnr[:, w, :] + (new_cnr[:, w, :] - new_cnr[:, w-1, :])

            return new_cnr

        def project_square(img, x, y):
            """
            4-Point deskew using Perspective Transform
            """
            pts1 = np.float32([
                corners[x, y],
                corners[x, y+1],
                corners[x+1, y],
                corners[x+1, y+1],
            ])
            pts2 = np.float32([
                [0, 0],
                [SQUARE_WIDTH, 0],
                [0, SQUARE_WIDTH],
                [SQUARE_WIDTH, SQUARE_WIDTH],
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            sq_img = cv2.warpPerspective(img, M, (SQUARE_WIDTH, SQUARE_WIDTH))

            return sq_img

        # impute outer corners
        corners = impute_corners(corners)

        # dimensions of board image
        w = h = SQUARE_WIDTH * CHESSBOARD_WIDTH
        board_img = np.zeros((w, h, img.shape[2]), np.uint8)

        # project each of the squares
        for x in range(CHESSBOARD_WIDTH):
            for y in range(CHESSBOARD_WIDTH):
                board_img[
                    (x * SQUARE_WIDTH):((x + 1) * SQUARE_WIDTH),
                    (y * SQUARE_WIDTH):((y + 1) * SQUARE_WIDTH),
                    :
                ] = project_square(img, x, y)

        return board_img

    def get_chessboard_corners(self, img, draw_corners=False):
            """
            From OpenCV documentation
            """
            w = CHESSBOARD_WIDTH - 1 # OpenCV alg only detects inner corners

            # termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # find the corners
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (w, w), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                cnr = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Draw and display the corners
                if draw_corners:
                    img = cv2.drawChessboardCorners(img, (w, w), cnr, ret)

                return cnr, img
            else:
                return corners, img

    def calibrate_pieces(self, calib_img):
        def get_square_image(board_img, sq):
            """
            Outputs the square image based on square index {0...63}
            """
            x_offset = CHESSBOARD_WIDTH - 1 - int(sq / CHESSBOARD_WIDTH)
            y_offset = (sq % CHESSBOARD_WIDTH)
            x = x_offset * SQUARE_WIDTH
            y = y_offset * SQUARE_WIDTH

            return board_img[x:x+SQUARE_WIDTH, y:y+SQUARE_WIDTH, :]

        def piece_edge_scores(sq_img):
            """
            Returns a list of scores + a masked image used in scores from a square image
            """
            def get_masked_circle(img):
                try:
                    h, w, _ = img.shape
                except:
                    h, w = img.shape
                circle_img = np.zeros((h, w), np.uint8)
                cv2.circle(circle_img, (int(w / 2), int(h / 2)), int(SQUARE_WIDTH / 3), 1, thickness=-1)
                masked_img = cv2.bitwise_and(img, img, mask=circle_img)
                return masked_img

            def apply_threshold(img):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.GaussianBlur(img, (5, 5), 0)
                img = cv2.adaptiveThreshold(
                    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                return img

            # create copies with transformations / masks
            masked = get_masked_circle(sq_img)
            bw_sq_img = apply_threshold(sq_img)
            bw_m_sq_img = get_masked_circle(bw_sq_img)

            # contour score 1 - BW
            contours, _ = cv2.findContours(bw_sq_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            c1_score = sum([len(x.flatten()) for x in contours])

            # contour score 2 - circle masked BW
            contours, _ = cv2.findContours(bw_m_sq_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            c2_score = sum([len(x.flatten()) for x in contours])

            return [
                np.mean(cv2.Canny(masked, 50, 100, 3)), # circle masked RGB
                np.mean(cv2.Canny(bw_m_sq_img, 50, 100, 3)), # circle masked BW
                np.mean(cv2.Canny(masked, 100, 150, 3)), # circle masked RGB
                np.mean(cv2.Canny(bw_m_sq_img, 100, 150, 3)), # circle masked BW
                c1_score,
                c2_score
            ], masked

        def fit_piece_presence(board_img):
            """
            Build model for detecting whether a square has {
                1 - any piece
                0 - no piece
            }
            Assumes board_img is an calibrated image at starting position
            """
            # default calibration layout
            labels = [int(x) for x in '1' * 16 + '0' * 32 + '1' * 16]

            # features
            X = []
            for sq in range(CHESSBOARD_SQUARES):
                sq_img = get_square_image(board_img, sq)
                ss, _ = piece_edge_scores(sq_img)
                X.append(ss)

            # standardise
            scaler = StandardScaler()
            scaler = scaler.fit(X)
            X_scaled = scaler.transform(X)

            # model
            model = LogisticRegression()
            model = model.fit(X_scaled, labels)

            # performance
            #predicted = model.predict(X_scaled)
            #print(metrics.confusion_matrix(labels, predicted))
            #print(metrics.classification_report(labels, predicted))

            return scaler, model

        def radial_gray_hist(sq_img):
            """
            Apply centre-weighted radial kernel and calculate grayscale histogram
            """
            # must be a square
            h, w, _ = sq_img.shape
            if h != w:
                raise Exception('Square image does not have h = w')

            # centre radial weights
            rad_mat = np.zeros((h, w), np.float32)
            for x in range(h):
                for y in range(w):
                    rad_mat[x, y] = ((h - abs(h - x - x - 1)) / h) * ((w - abs(w - y - y - 1)) / w)

            # make everything outside radius / 3 -> 0 weight
            circle_img = np.zeros((h, w), np.uint8)
            cv2.circle(circle_img, (int(w / 2), int(h / 2)), int(SQUARE_WIDTH / 3), 1, thickness=-1)
            rad_mat = cv2.bitwise_and(rad_mat, rad_mat, mask=circle_img)

            # calculated grayscale histogram and gray value mode
            sq_img = cv2.cvtColor(sq_img, cv2.COLOR_BGR2GRAY)
            hist = [rad_mat.reshape(h*w)[sq_img.reshape(h*w) == group].sum() for group in range(256)]

            return hist

        def fit_piece_colours(board_img):
            """
            Build model for detecting whether a square has {
                1 - player 1 piece
                2 - player 2 piece
                0 - no piece
            }
            Assumes board_img is an calibrated image at starting position
            """
            # default calibration layout
            #labels = [x for x in '1' * 16 + '0' * 32 + '2' * 16]
            labels1 = labels2 = [x for x in '1' * 8 + '2' * 8]

            # define feature signature for each square
            X1 = []
            X2 = []
            for sq in range(CHESSBOARD_SQUARES):
                # get feature signature for square
                sq_img = get_square_image(board_img, sq)
                hist = radial_gray_hist(sq_img)
                #cv2.imwrite('./calibration/sq_{0}.jpg'.format(sq), sq_img)

                # prepare 2 datasets for alternating squares
                if not 15 < sq < 48:
                    if sq % 2 == (sq // 8) % 2:
                        X1.append(hist)
                    else:
                        X2.append(hist)

            # Fit NN for 2 models only for piece colour, excludes no-piece
            # TODO: tune k instead of hard-code k
            neigh1 = KNeighborsClassifier(n_neighbors=3)
            neigh1.fit(X1, labels1)
            neigh2 = KNeighborsClassifier(n_neighbors=3)
            neigh2.fit(X2, labels2)

            return (neigh1, neigh2)

        def fit_performance(board_img, piece_model, presence_model):
            def square_has_piece(sq_img, presence_model):
                """
                Returns True / False on whether a square image has a piece in it (any colour)
                """
                scores, x_img = piece_edge_scores(sq_img)

                # hardcoded thresholds
                #has_piece = (scores[0] > 6) + (scores[1] > 8) + (scores[2] > 6) + (scores[3] > 8) + (scores[4] > 300) > 2

                # model
                scaler, model = presence_model
                has_piece = model.predict(scaler.transform(np.array(scores, np.float64).reshape(1, -1))) == 1

                return has_piece, scores, x_img

            def format_squares_piece_text(sq_pc):
                """
                Pretty print board layout
                """
                i = 0
                sq_pc_formatted = []
                for pc in sq_pc:
                    i += 1
                    sq_pc_formatted.append(pc)
                    sq_pc_formatted.append(' ')
                    if i % 8 == 0:
                        sq_pc_formatted.append('\n')

                return ''.join(sq_pc_formatted)

            '''
            Measure performance - cheat and show predicted values on training set
            '''
            neigh1, neigh2 = piece_model

            squares = []
            for sq in range(CHESSBOARD_SQUARES):
                sq_img = get_square_image(board_img, sq)
                hist = radial_gray_hist(sq_img)
                hist = np.array(hist).reshape(1, -1)
                shp, ss, test_img = square_has_piece(sq_img, presence_model)
                print(sq, [int(s) for s in ss])
                #cv2.imwrite('./calibration/m_{0}.jpg'.format(sq), test_img)
                if shp:
                    if sq % 2 == (sq // 8) % 2:
                        squares.append(neigh1.predict(hist)[0])
                    else:
                        squares.append(neigh2.predict(hist)[0])
                else:
                    squares.append('0')
            format_squares_piece_text(squares)
            #print('Current predicted values', ''.join(squares))
            chunks = [squares[x:x+8] for x in range(0, len(squares), 8)]
            return chunks

        # 2nd calibration image - starting board
        # load corners
        try:
            #corners = np.load('./calibration/corners.npy')
            corners=self.board_corners
        except:
            raise Exception('Calibration step 1 not run yet')

        # train piece detection model as save to file
        calib_img2 = self.project_board(calib_img, corners=corners)
        presence_model = fit_piece_presence(calib_img2)
        piece_model = fit_piece_colours(calib_img2)

        # performance
        result=fit_performance(calib_img2, piece_model, presence_model)

        #with open('./calibration/piece_model.pkl', 'wb') as f:
            #pickle.dump(piece_model, f)
        #with open('./calibration/presence_model.pkl', 'wb') as f:
            #pickle.dump(presence_model, f)
        print('Piece model trained and saved')

        # save second board calibration image
        #cv2.imwrite('./calibration/starting-board-projected.jpg', calib_img2)

        print('Calibration step 2 completed')
        return result

    def game_checking(self, bmp, player):
        detected_move=""
        # decode from base64
        img = self.decodeImage(bmp)

        if(self.board_matrix == None):
            self.board_matrix = self.calibrate_pieces(img)
            #self.board_matrix=[ ["1","1","1","1","1","1","1","1"],
                            #["1","1","1","1","1","1","1","1"],
              #              ["0","0","0","0","0","0","0","0"],
               #              ["0","0","0","0","0","0","0","0"],
               #              ["0","0","0","0","0","0","0","0"],
               #              ["0","0","0","0","0","0","0","0"],
               #              ["2","2","2","2","2","2","2","2"],
               #              ["2","2","2","2","2","2","2","2"]]

            detected_move = "START"
            print(self.board_matrix)
            return detected_move
        else:
            if (player == "White"):
                turn = chess.WHITE
            elif (player == "Black"):
                turn = chess.BLACK
            else:
                detected_move = "You escaped the simulation"
                print(detected_move)
                return detected_move
            
            current_board = self.calibrate_pieces(img)
            legal_moves = [x.uci() for x in self.board.legal_moves]
            print(self.board)            
            start=None
            end=None
            old_pos = []
            new_pos = []
            
            #castling
            casteled=False
            if(self.board.has_castling_rights(turn)):
                print("Checking castling move...")
                if(turn==chess.WHITE):
                    print("Checking castling move white...")
                    r=0
                elif(turn==chess.BLACK):
                    print("Checking castling move black...")
                    r=7
                if(current_board[r][ord('e')-97]=="0"):
                    print("RE mosso")
                    if(current_board[r][ord('h')-97]=="0" and self.board_matrix[r][ord('h')-97]!="0"):
                        detected_move="e"+str(r+1)+"g"+str(r+1)
                        # double update, how?
                        print("Arrocco corto...")
                        casteled=True
                    elif(current_board[r][ord('a')-97]=="0" and self.board_matrix[r][ord('a')-97]!="0"):
                        detected_move="e"+str(r+1)+"c"+str(r+1)
                        print("Arrocco lungo...")
                        casteled=True

            #normal-move
            if(not casteled):
                for row in range(8): #numeri
                    for column in range(8): #lettere
                        if(current_board[row][column]!=self.board_matrix[row][column]):
                            print(current_board[row][column]+" "+self.board_matrix[row][column])
                            if(current_board[row][column]=="0"):
                                old_pos = [row, column]
                                start=chr(97+column)+str(row+1)
                                print(start)
                            else:
                                new_pos = [row, column]
                                end=chr(97+column)+str(row+1)
                                print(end)
                if(start==None or end==None):
                    print("No moves detected")
                else:
                    detected_move=start+end
            if(detected_move==None):
                detected_move = "No moves detected"
                print(detected_move)
            elif((detected_move) in legal_moves):
                print("Move detected: "+detected_move)
                move = chess.Move.from_uci(detected_move)
                self.board.push(move)
                #self.board_matrix = current_board
                #detected_move=start+"/"+end
                self.updateBoardMatrix(player, old_pos, new_pos)

            else:
                wrongMoves = detected_move
                detected_move = f'Illegal Move: mossa riconosciuta= {wrongMoves}; armetti a posto'

                print(detected_move)


        return detected_move

    def decodeImage(self, bitmap):
        decoded_image= base64.b64decode(bitmap)
        image= np.fromstring(decoded_image,np.uint8)
        image_as_ndarray= cv2.imdecode(image,cv2.IMREAD_UNCHANGED)
        image_without_alpha_channel= image_as_ndarray[:,:,:3]
        return image_without_alpha_channel

    def encodeImage(self, img):
        pil_im = Image.fromarray(img)
        buff = io.BytesIO()
        pil_im.save(buff,format="PNG")
        img_str = base64.b64encode(buff.getvalue())
        return "" + str(img_str, 'utf-8')


    # def updateBoardMatrix(self, player , detected_move):
    #     pos = detected_move.split("/")
    #     # ord(column) - 97 -> column number
    #     # row - 1 -> row number
    #     old_pos = pos[0]
    #     new_pos = pos[1]
    #     old_col = ord(old_pos[0]) - 97
    #     old_row = old_pos[1] - 1
    #     new_col = ord(new_pos[0]) - 97
    #     new_row = new_pos[1] - 1 
    #     self.board_matrix[old_row][old_col] = "0"
    #     if (player == "White"):
    #         self.board_matrix[new_row][new_col] = "1"
    #     elif (player == "Black"):
    #         self.board_matrix[new_row][new_col] = "2"    

    def updateBoardMatrix(self, player , old_pos, new_pos):
        self.board_matrix[old_pos[0]][old_pos[1]] = "0"
        if (player == "White"):
            self.board_matrix[new_pos[0]][new_pos[1]] = "1"
        elif (player == "Black"):
            self.board_matrix[new_pos[0]][new_pos[1]] = "2"    
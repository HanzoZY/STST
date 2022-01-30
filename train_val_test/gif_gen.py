import imageio
import os
def create_gif(image_list, gif_name, duration = 1.0):
    '''
    生成GIF
    :param image_list:
    :param gif_name:
    :param duration:
    :return:
    '''
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def get_all_file(file_dir, tail_list=('.jpg', '.png', '.jpeg')):
    """
    获取所有的文件名
    :param file_dir:    指定目录
    :param tail_list:   指定文件类型（后缀名）
    :return:
    """
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            for tail in tail_list:
                if file.endswith(tail):
                    file_list.append(os.path.join(root, file))
                    break
    file_list.sort() # 排序
    return file_list

if __name__ == '__main__':


    # 指定输入目录， 与输出目录
    input_dir, output_dir = './skeleton_sequence/', './skeleton_sequence/'
    # 每张图片停留的时间（秒）
    duration = 0.2


    file_list = get_all_file(input_dir)
    create_gif(file_list, os.path.join(output_dir, 'result_c.gif'), duration)
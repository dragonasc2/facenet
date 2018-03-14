import pymysql
from DBUtils.PooledDB import PooledDB
import datetime

server_ip = '**'
user_name = '**'
password = '**'
schema = 'face_recognition'
pool = PooledDB(pymysql, 50, host=server_ip, user=user_name, passwd=password, db=schema,
                use_unicode=True, charset='utf8')
execution_log_table_name = 'execution_log'


def get_connection():
    """
    从MySQL连接池中获得一个数据库链接
    Returns:
        一个数据库连接
    """
    return pool.connection()


class ExecutionLogger:
    """记录程序关键段运行时间的工具
    在构造函数执行时，记录起始时间并写入数据库；在log_finish执行时，记录结束时间以及其他信息，并写入数据库

    使用方法：
    logger = ExecutionLogger('some operation time')
    Your time consuming code...
    ...
    ...
    logger.log_finish(image_height=1920, image_width=1080, info='large CNN)
    """
    def __init__(self, operation_name):
        """
        建立数据库连接，并记录开始时间信息
        Args:
            operation_name: str, 所记录的操作的名字
        """
        self._conn = get_connection()
        self.connection_avaible = True
        self._start_logged = False
        self._finish_logged = False
        self._log_start(operation_name)

    def _log_start(self, operation_name):
        """
        记录开始时间信息，并写入数据库
        Args:
            operation_name: str, 所记录的操作的名字
        Returns:
            创建的数据库记录的record_id
        """
        if operation_name is None:
            raise ValueError('必须指定operation_name')
        if self._start_logged:
            return
        self._start_logged = True
        self._start_time = datetime.datetime.now()
        cursor = self._conn.cursor()
        cursor.execute('INSERT INTO '+ execution_log_table_name + ' (operation_name, start_time) values (%s, now())',
                       (operation_name, ))
        self._conn.commit()
        cursor.execute('SELECT LAST_INSERT_ID()')
        results = cursor.fetchone()
        self._record_id = results[0]
        cursor.close()
        return self._record_id

    def log_finish(self, image_height=-1, image_width=-1, info=''):
        """
        记录结束时间，计算耗时。然后将耗时和其他信息写入数据库
        Args:
            image_height: 所处理的图像的高
            image_width: 所处理的图像的宽
            info: 额外的信息
        """
        if self._finish_logged or not self.connection_avaible:
            return
        self._finish_logged = True
        self._finish_time = datetime.datetime.now()
        cursor = self._conn.cursor()
        cursor.execute('UPDATE ' + execution_log_table_name + ' SET '
                       'image_height=%s, '
                       'image_width=%s, '
                       'time_cost_in_second=%s,'
                       'info=%s '
                       'WHERE execution_log_id=%s',
                       (image_height,
                        image_width,
                        (self._finish_time - self._start_time).total_seconds(),
                        info,
                        self._record_id
                        ))
        self._conn.commit()
        cursor.close()
        self._conn.close()

    def close(self):
        if self._conn:
            self._conn.close()
        self.connection_avaible = False

    def __del__(self):
        """
        释放所有资源
        """
        self.connection_avaible = False
        try:
            self._conn.close()
        except:
            pass





if __name__ == '__main__':
    A = get_connection()
    A.close()
    print('main running')
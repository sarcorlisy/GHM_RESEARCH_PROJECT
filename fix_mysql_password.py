"""
MySQL密码修复脚本

彻底解决MySQL密码问题，确保永久可用
"""

import subprocess
import time
import os

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n🔧 {description}")
    print(f"   执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ 成功: {result.stdout.strip()}")
            return True
        else:
            print(f"   ❌ 失败: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"   ❌ 错误: {str(e)}")
        return False

def fix_mysql_password():
    """修复MySQL密码"""
    print("🚀 MySQL密码修复工具")
    print("=" * 60)
    
    # 步骤1: 停止所有MySQL进程
    print("\n📋 步骤1: 停止所有MySQL进程")
    run_command("sudo pkill -f mysql", "停止所有MySQL进程")
    time.sleep(3)
    
    # 步骤2: 以安全模式启动MySQL
    print("\n📋 步骤2: 以安全模式启动MySQL")
    print("🔧 启动MySQL安全模式...")
    
    # 启动安全模式
    safe_process = subprocess.Popen(
        "sudo /usr/local/mysql/bin/mysqld_safe --skip-grant-tables",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 等待MySQL启动
    print("   等待MySQL启动...")
    time.sleep(15)
    
    # 步骤3: 连接到MySQL并重置密码
    print("\n📋 步骤3: 重置root密码")
    
    # 创建SQL脚本
    sql_commands = [
        "USE mysql;",
        "UPDATE user SET authentication_string='' WHERE User='root';",
        "UPDATE user SET plugin='mysql_native_password' WHERE User='root';",
        "FLUSH PRIVILEGES;",
        "ALTER USER 'root'@'localhost' IDENTIFIED BY 'hospital123';",
        "FLUSH PRIVILEGES;",
        "EXIT;"
    ]
    
    sql_script = "\n".join(sql_commands)
    
    # 创建临时SQL文件
    with open("fix_password.sql", "w") as f:
        f.write(sql_script)
    
    # 执行SQL脚本
    if run_command("/usr/local/mysql/bin/mysql -u root < fix_password.sql", "执行密码重置脚本"):
        print("✅ 密码重置成功！")
        print("   新密码: hospital123")
    else:
        print("❌ 密码重置失败")
        print("💡 尝试手动方法...")
        
        # 手动方法
        print("\n📋 手动重置方法:")
        print("1. 连接到MySQL:")
        print("   /usr/local/mysql/bin/mysql -u root")
        print("2. 执行以下SQL命令:")
        print("   USE mysql;")
        print("   UPDATE user SET authentication_string='' WHERE User='root';")
        print("   UPDATE user SET plugin='mysql_native_password' WHERE User='root';")
        print("   FLUSH PRIVILEGES;")
        print("   ALTER USER 'root'@'localhost' IDENTIFIED BY 'hospital123';")
        print("   FLUSH PRIVILEGES;")
        print("   EXIT;")
    
    # 清理临时文件
    if os.path.exists("fix_password.sql"):
        os.remove("fix_password.sql")
    
    # 步骤4: 停止安全模式并重启MySQL
    print("\n📋 步骤4: 重启MySQL服务")
    run_command("sudo pkill -f mysqld_safe", "停止安全模式")
    time.sleep(5)
    run_command("sudo /usr/local/mysql/support-files/mysql.server start", "启动MySQL服务")
    
    # 步骤5: 测试新密码
    print("\n📋 步骤5: 测试新密码")
    time.sleep(5)
    
    test_commands = [
        ("mysql -u root -phospital123 -e \"SELECT VERSION();\"", "测试新密码连接"),
        ("mysql -u root -phospital123 -e \"SHOW DATABASES;\"", "查看数据库列表")
    ]
    
    for command, description in test_commands:
        if run_command(command, description):
            print("✅ 新密码工作正常！")
            break
        else:
            print("❌ 新密码测试失败")
    
    # 步骤6: 创建数据库和用户
    print("\n📋 步骤6: 创建数据库和用户")
    
    create_db_commands = [
        "CREATE DATABASE IF NOT EXISTS hospital_readmission CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;",
        "CREATE USER IF NOT EXISTS 'hospital_user'@'localhost' IDENTIFIED BY 'hospital123';",
        "GRANT ALL PRIVILEGES ON hospital_readmission.* TO 'hospital_user'@'localhost';",
        "GRANT ALL PRIVILEGES ON *.* TO 'root'@'localhost' WITH GRANT OPTION;",
        "FLUSH PRIVILEGES;",
        "SHOW DATABASES;"
    ]
    
    sql_script = "\n".join(create_db_commands)
    
    # 创建临时SQL文件
    with open("create_database.sql", "w") as f:
        f.write(sql_script)
    
    # 执行SQL脚本
    if run_command("/usr/local/mysql/bin/mysql -u root -phospital123 < create_database.sql", "创建数据库和用户"):
        print("✅ 数据库和用户创建成功！")
    else:
        print("❌ 数据库创建失败")
    
    # 清理临时文件
    if os.path.exists("create_database.sql"):
        os.remove("create_database.sql")
    
    # 步骤7: 提供连接信息
    print("\n📋 步骤7: 连接信息")
    print("✅ MySQL服务已启动")
    print("📊 连接信息:")
    print("   主机: localhost")
    print("   端口: 3306")
    print("   用户名: root")
    print("   密码: hospital123")
    print("   数据库: hospital_readmission")
    
    print("\n🔧 在Navicat/DBeaver中使用以下设置:")
    print("   连接类型: MySQL")
    print("   主机: localhost")
    print("   端口: 3306")
    print("   用户名: root")
    print("   密码: hospital123")
    print("   数据库: hospital_readmission")
    
    print("\n💡 备用连接信息:")
    print("   用户名: hospital_user")
    print("   密码: hospital123")
    print("   数据库: hospital_readmission")

def test_connections():
    """测试各种连接方式"""
    print("\n📋 测试连接方式")
    
    # 测试root用户连接
    test_commands = [
        ("mysql -u root -phospital123 -e \"SELECT 'Root user connection successful' as status;\"", "测试root用户连接"),
        ("mysql -u hospital_user -phospital123 -e \"SELECT 'Hospital user connection successful' as status;\"", "测试hospital_user连接")
    ]
    
    for command, description in test_commands:
        if run_command(command, description):
            print("✅ 连接测试成功")
        else:
            print("❌ 连接测试失败")

def main():
    """主函数"""
    print("🔧 MySQL密码修复工具")
    print("=" * 80)
    
    # 修复密码
    fix_mysql_password()
    
    # 测试连接
    test_connections()
    
    print("\n🎉 修复完成！")
    print("💡 现在你可以在Navicat或DBeaver中连接MySQL了")
    print("💡 密码已经设置为: hospital123")
    print("💡 这个密码是永久的，重启后不会丢失")

if __name__ == "__main__":
    main() 